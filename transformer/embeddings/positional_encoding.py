import torch
from torch import nn

# 클래스 정의
class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(positional_encoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(1,max_len, device=device).float().unsqueeze(dim=1)

        self.encoding[: , 0::2] = torch.sin(pos / 10000 ** (2* torch.arange(0,d_model , step=2 , device=device).float()))
        self.encoding[: , 1::2] = torch.cos(pos / 10000 ** (2* torch.arange(0,d_model , step=2 , device=device).float()))

    def forward(self, x):
        batch_size , seq_len = x.size() # 한 번에 처리할 문장 
        return self.encoding[ : seq_len, :] 

