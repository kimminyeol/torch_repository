import torch
import torch.nn as nn
from layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_head):
        self.num_head = num_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def forward(self,q,k,v,mask=None):

        q,k,v= self.w_q(q), self.w_k(k), self.w_v(v)
        q,k,v = self.split(q), self.split(k), self.split(v)
        out , attention = self.attention(q,k,v,mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out 
    

    def split(self, tensor):
        batch_size, seq_len, d_model = tensor.size()
        d_tensor = d_model // self.num_head
        tensor = tensor.view(batch_size, seq_len, self.num_head, d_tensor).transpose(1,2)
        return tensor

    def concat(self, tensor):
        batch_size, num_head, seq_len, d_tensor = tensor.size()
        # contiguous() 메소드는 텐서를 메모리에 연속적으로 배치시킨다.
        # 차원 변환 순서 
        # 1. (batch_size, num_head, seq_len, d_tensor) -> (batch_size, seq_len, num_head, d_tensor)
        # 2. (batch_size, seq_len, num_head, d_tensor) -> (batch_size, seq_len, num_head*d_tensor)
        tensor = tensor.transpose(1,2).contiguous().view(batch_size, seq_len, num_head*d_tensor)
        return tensor