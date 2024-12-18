import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import config, set_seed
from dataset import CustomDataset
from model import CustomEfficientFormerWithShallowNN
from train_utils import train_one_epoch, evaluate, test
from plot_utils import plot_metrics

def main():
    # 시드 설정
    set_seed(config['training']['random_seed'])

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")

    # 데이터 로드 및 분할
    dataset = datasets.ImageFolder(root=config['data']['image_root'])
    image_paths, labels = zip(*dataset.samples)
    additional_data = torch.load(config['data']['additional_data_path'])
    features, feature_labels = additional_data

    num_images = len(image_paths)
    num_features = features.shape[0]

    # 이미지와 추가 피처의 개수 일치 여부 확인
    if num_images != num_features:
        print(f"경고: 이미지 수({num_images})와 추가 피처 수({num_features})가 일치하지 않습니다.")
        min_samples = min(num_images, num_features)
        print(f"처음 {min_samples}개의 샘플을 무작위로 선택하여 학습합니다.")

        # 무작위 인덱스 생성
        indices = np.random.permutation(min_samples)

        # 데이터 샘플링
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
        features = features[indices]
        feature_labels = feature_labels[indices]
    else:
        print("이미지 수와 추가 피처 수가 일치합니다.")

    # 추가 피처 정규화
    features = StandardScaler().fit_transform(features)

    # 데이터 분할: Train (70%), Validation (15%), Test (15%)
    train_imgs, temp_imgs, train_feats, temp_feats, train_labels, temp_labels = train_test_split(
        image_paths, features, labels, test_size=0.3, stratify=labels, random_state=config['training']['random_seed']
    )
    val_imgs, test_imgs, val_feats, test_feats, val_labels, test_labels = train_test_split(
        temp_imgs, temp_feats, temp_labels, test_size=0.5, stratify=temp_labels, random_state=config['training']['random_seed']
    )

    print(f"Train 샘플 수: {len(train_imgs)}")
    print(f"Validation 샘플 수: {len(val_imgs)}")
    print(f"Test 샘플 수: {len(test_imgs)}")

    # 데이터 증강
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    train_dataset = CustomDataset(train_imgs, train_feats, train_labels, transform=transform_train)
    val_dataset = CustomDataset(val_imgs, val_feats, val_labels, transform=transform_val_test)
    test_dataset = CustomDataset(test_imgs, test_feats, test_labels, transform=transform_val_test)

    # WeightedRandomSampler 사용 (Train 데이터에만 적용)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size']['train'], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size']['val'])
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size']['test'])

    # 모델 및 옵티마이저 설정
    model = CustomEfficientFormerWithShallowNN(
        additional_feature_dim=config['data']['additional_feature_dim'],
        num_classes=config['data']['num_classes']
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['initial_lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler']['cosine_t0'], T_mult=config['scheduler']['cosine_tmult'])
    scaler = GradScaler()

    # 학습 루프 초기화
    best_val_loss = float('inf')
    no_improvement = 0  # Early Stopping을 위한 변수 초기화
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, config['training']['gradient_accumulation_steps'])
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc * 100)  # % 단위로 변환
        val_accs.append(val_acc * 100)      # % 단위로 변환

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        # Early Stopping 로직 추가
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])
            print("모델이 저장되었습니다.")
        else:
            no_improvement += 1
            print(f"검증 손실이 개선되지 않았습니다. (개선 없음: {no_improvement}/{config['training']['patience']})")
            if no_improvement >= config['training']['patience']:
                print("조기 종료가 트리거되었습니다.")
                break
       
    # 최적의 모델 로드
    model.load_state_dict(torch.load(config['MODEL_SAVE_PATH']))
    print("최적의 모델이 로드되었습니다.")

    # 손실 및 정확도 추세 시각화
    plot_metrics(train_losses, val_losses, train_accs, val_accs, epoch + 1)

    # 테스트 데이터 평가
    test(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()
