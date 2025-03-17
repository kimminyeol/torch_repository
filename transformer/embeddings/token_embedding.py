import torch.nn as nn
import torch

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        # super을 활용함으로써, nn.embedding 호출 
        super(TokenEmbedding, self).__init__(vocab_size,d_model,padding_idx=1) 
        # padding_idx=1 을 설정함으로써 입력 값이 1인 모든 위치의 임베딩 벡터가 항상 0 벡터로 고정된다. 
        # 패딩 토큰을 1로 인식하여 0으로 바꿔버린다. 
        # 즉, 패딩 위치의 가중치는 전혀 학습되지 않는다. 

