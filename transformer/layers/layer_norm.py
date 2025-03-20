import torch 
import torch.nn as nn   

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-12):
        super(LayerNorm, self).__init__()
        # 학습을 통해 감마와 베타를 조절하여 레이어 정규화의 효과를 조절할 수 있기 때문
        self.gamma = self.Parameter(torch.ones(d_model)) # 
        self.beta = self.Parameter(torch.zeros(d_model))
        self.eps = eps # epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # -1을 사용하면 마지막 차원을 기준으로 평균과 표준편차를 구함
        out = (x-mean)/ (std + self.eps)
        out = self.gamma * out + self.beta
        return out