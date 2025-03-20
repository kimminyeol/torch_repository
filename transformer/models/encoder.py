import torch.nn as nn
from blocks.encoder_layer import EncoderLayer
from embeddings.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_vec_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob , device):
        super.__init__()
        self.embedding = TransformerEmbedding(vocab_size=enc_vec_size, d_model=d_model, max_len=max_len, drop_prob=drop_prob ,device=device)
        self.layers = nn.Modulelist([EncoderLayer(d_model=d_model, num_head=n_head, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
                                        for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x= self.embedding(x)

        for layer in self.layers:
            x= layer(x, src_mask)
        
        return x 
