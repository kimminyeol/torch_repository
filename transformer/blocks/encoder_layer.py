from torch import nn

# 셀프 어텐션 -> 배치정규화+피드포워드 신경망 -> 배치정규화+잔차연결

from layers.multi_head_attention import MultiHeadAttention
from layer.normalization import LayerNormalization
from layers.position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_head,hidden, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
    
    def forward(self, x, mask):
        _x= x
        x= self.attention(x,x,x,mask)
        x= self.dropout1(x)
        x= self.norm1(x+_x)
        _x= x
        x = self.ffn(x)
        x= self.dropout2(x)
        x= self.norm2(x+_x)
        return x