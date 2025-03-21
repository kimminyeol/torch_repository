import torch
from torch.nn import functional as F 

class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dataloader, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.device = device
    
    def train(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            print(f"🔥 Epoch {epoch + 1}/{epochs} 시작...")

            for step, batch in enumerate(self.train_dataloader):
                # 🔹 Batch 출력 (디버깅 용)
                if step == 0:
                    print("🔍 Batch 구조 확인:", batch)

                src = batch['input_ids'].to(self.device)
                trg = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 🔹 Forward & Backward
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=src, attention_mask=attention_mask, labels=trg)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # 🔹 진행 상황 출력
                if step % 10 == 0:  
                    print(f"🟣 Epoch {epoch + 1} | Step {step} | Loss: {loss.item():.4f}")

            # 🔹 Epoch 종료 후 평균 Loss 출력
            avg_loss = total_loss / len(self.train_dataloader)
            print(f"✅ Epoch {epoch + 1} 완료 | Average Loss: {avg_loss:.4f}")