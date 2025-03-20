import torch.nn as nn
from layers.FFNN import FFNN
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self,hidden, d_model, n_head, drop_pob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(drop_pob=drop_pob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(drop_pob=drop_pob)
        
        self.feedforward = FFNN(d_model=d_model, hidden=hidden, drop_prob=drop_pob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(drop_pob=drop_pob)
    
    def forward(self, d, enc, trg_mask, src_mask):
        _x = d
        x= self.self_attention(d,d,d,trg_mask)
        
        # add and norm
        x= self.dropout1(x)
        x += _x
        x= self.norm1(x)

        # cross-attention
        _x = x 
        x= self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

        # add and norm
        x= self.dropout2(x)
        x= self.norm2(x+_x)

        # FFNN
        _x= x
        x= self.feedforward(x)

        # add and norm
        x= self.dropout3(x)
        x= self.norm3(x + _x)
        return x 


        
