import torch
from model.resnet import ResNet
from model.residual_block import ResidualBlock
from data.data_loader import get_data_loaders
from model.base_model import evaluate_model

# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet(num_classes=10, block=ResidualBlock, layers=[2, 2, 2, 2]).to(device)

# 저장된 모델 로드
model.load_state_dict(torch.load('resnet18_cifar10.pth'))

# 데이터 로드
_, test_loader = get_data_loaders(batch_size=128)

# 평가
evaluate_model(model, test_loader, device)
