import torch.optim as optim

def get_optimizer(model, learning_rate=5e-5, weight_decay=0.01):
    return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)