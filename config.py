import random
import numpy as np
import torch

# ----------------------------
# 설정 및 하이퍼파라미터
# ----------------------------
config = {
    'data': {
        'image_root': 'art/data/images',  # 이미지 데이터 경로
        'additional_data_path': 'art/data/run_length_features_parallel.pt',  # 추가 피처 데이터 경로
        'batch_size': {
            'train': 32,
            'val': 32,
            'test': 32
        },
        'additional_feature_dim': 20,  # 추가 피처 차원
        'num_classes': 49  # 분류 클래스 수
    },
    'training': {
        'initial_lr': 0.001,
        'num_epochs': 20,
        'patience': 3,  # 조기 종료를 위한 인내심 
        'random_seed': 42,  # 랜덤 시드
        'gradient_accumulation_steps': 4  # Gradient Accumulation
    },
    'scheduler': {
        'cosine_t0': 10,  # CosineAnnealingWarmRestarts 초기 주기
        'cosine_tmult': 2  # 주기 증가 계수
    },
    'MODEL_SAVE_PATH': 'art/output/best_model.pth'  # 모델 저장 경로
}

# ----------------------------
# 재현성을 위한 시드 설정
# ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
