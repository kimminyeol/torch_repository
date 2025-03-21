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
            for src, trg in self.train_dataloader:
                src , trg= src.to(self.device) , trg.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids=src, labels=trg)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")