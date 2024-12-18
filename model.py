import torch
import torch.nn as nn
from timm import create_model

class CustomEfficientFormerWithShallowNN(nn.Module):
    def __init__(self, additional_feature_dim: int, num_classes: int):
        super(CustomEfficientFormerWithShallowNN, self).__init__()

        # EfficientFormer-L1 모델 로드
        self.base_model = create_model(
            'efficientformer_l1',
            pretrained=True,
            num_classes=0  # 분류 헤드 제거
        )

        # EfficientFormer의 모든 헤드 비활성화
        if hasattr(self.base_model, 'classifier'):
            self.base_model.classifier = nn.Identity()

        if hasattr(self.base_model, 'head'):
            self.base_model.head = nn.Identity()

        # Shallow Neural Network (얕은 신경망) 정의
        self.shallow_nn = nn.Sequential(
            nn.Linear(additional_feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # 이미지와 추가 피처를 결합한 후 최종 분류기 정의
        self.classifier = nn.Sequential(
            nn.Linear(256 + self.base_model.num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, additional_features):
        # EfficientFormer로부터 이미지 특징 추출
        image_features = self.base_model(images)  # [Batch, num_features]

        # 추가 피처를 얕은 신경망에 통과
        additional_features_embedded = self.shallow_nn(additional_features)

        # 이미지와 추가 피처 결합
        combined_features = torch.cat((image_features, additional_features_embedded), dim=1)  # [Batch, num_features + 256]

        # 최종 분류기 통과
        logits = self.classifier(combined_features)
        return logits
