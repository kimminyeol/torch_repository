import torch

def save_checkpoint(model, optimizer, epoch, path="checkpoints/model_checkpoint.pth"):
    """모델과 옵티마이저의 상태를 저장합니다."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path="checkpoints/model_checkpoint.pth"):
    """저장된 체크포인트를 로드하고 epoch을 반환합니다."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint['epoch']