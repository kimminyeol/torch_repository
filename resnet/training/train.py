
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model.resnet import ResNet
from model.residual_block import ResidualBlock
from model.bottleneck_block import BottleneckBlock
from data.data_loader import get_data_loaders
from model.base_model import train_model

# 하이퍼파라미터 설정
num_epochs = 25
batch_size = 128
learning_rate = 0.001

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 로드
    train_loader, test_loader = get_data_loaders(batch_size)
    dataloaders = {'train': train_loader, 'val': test_loader}
    print("🔍 Checking DataLoader...")
    for images, labels in train_loader:
        print(f"[INFO] Train Batch Size: {images.size(0)} | Image Shape: {images.shape} | Labels Shape: {labels.shape}")
        break

    # 모델 초기화
    model = ResNet(num_classes=10, block=ResidualBlock, layers=[2, 2, 2, 2]).to(device)

    # 손실 함수 및 최적화 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 학습
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)

    # 모델 저장
    torch.save(model.state_dict(), 'resnet18_cifar10.pth')