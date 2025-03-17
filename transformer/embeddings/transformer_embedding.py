from torch import nn 
from embeddings.positional_encoding import positional_encoding
from embeddings.token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len, drop_prob ,device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = positional_encoding(d_model=d_model, max_len=max_len, device=device)
        self.drop_out = nn.Dropout(float= drop_prob, inplace=False)
    
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        
        return self.drop_out(tok_emb+ pos_emb)
        