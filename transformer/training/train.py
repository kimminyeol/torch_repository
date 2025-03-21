import torch
from transformers import AutoModelForSeq2SeqLM
from data.dataloader import get_dataloader
from data.tokenizer import Tokenizer
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from training.trainer import Trainer
from training.optimizer import get_optimizer
from training.scheduler import get_scheduler
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 설정
set_seed(42)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 준비
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
tokenizer = Tokenizer(model_name='Helsinki-NLP/opus-mt-en-fr')

# 데이터 로드
train_dataloader = get_dataloader("en_fr_data.tsv", tokenizer, batch_size=64)

# 옵티마이저 및 스케줄러 설정
optimizer = get_optimizer(model)
scheduler = get_scheduler(optimizer, num_training_steps=len(train_dataloader) * 5)

# Trainer 초기화 및 학습 시작
trainer = Trainer(model, optimizer, scheduler, train_dataloader, device)

print("🔥 Training started...")
trainer.train(epochs=2)
print("✅ Training complete!")


# 체크포인트 저장
save_checkpoint(model, optimizer, epoch=5)