import torch.nn as nn
from blocks.decoder_layer import DecoderLayer
from embeddings.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_vec_size, max_len, d_model, ffn_hidden,n_head, n_layers, drop_prob, device):
        super.__init__()
        self.embedding = TransformerEmbedding(vocab_size=dec_vec_size, d_model=d_model, max_len=max_len, device=device, drop_prob=drop_prob)
        self.layers = nn.Modulelist([DecoderLayer(hidden=ffn_hidden, d_model=d_model, n_head=n_head, drop_pob=drop_prob) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_vec_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.embedding(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask , src_mask)
        
        output = self.linear(trg)
        return output