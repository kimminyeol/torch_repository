import torch
import torch.nn as nn

#
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, q, k, v, mask=None, e= 1e-12):
        batch_size, num_head, seq_len , d_k = k.size()

        k_t = k.transpose(2,3)
        score= torch.matmul(q,k_t) / torch.sqrt(d_k)
        # matmul과 @은 같은 기능을 수행한다.
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)
        
        score = self.softmax(score)

        v= score @ v
        return v, score
