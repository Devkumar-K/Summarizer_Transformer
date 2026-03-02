import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        
        if mask is not None:
           
            scores = scores.masked_fill(mask == 0, -1e9)

        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, Lq, D)

        return out, attn



class MultiHeadAttention(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None):
       
        B = query.size(0)

        
        Q = self.W_q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        
        attn_out, _ = self.attention(Q, K, V, mask)  

        
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        )

        
        return self.W_o(attn_out)



class PositionWiseFeedForward(nn.Module):
    

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        return self.fc2(self.dropout(F.relu(self.fc1(x))))



class PositionalEncoding(nn.Module):
   

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                             
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                              

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  

        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: (B, L, D) → (B, L, D)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)



class EncoderLayer(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
       
        
        sa = self.self_attn(x, x, x, src_mask)         
        x = self.norm1(x + self.drop1(sa))

        
        ff = self.ffn(x)
        x = self.norm2(x + self.drop2(ff))

        return x  # (B, S, D)



class DecoderLayer(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, cross_mask=None):
        
        
        sa = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(sa))

        
        ca = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = self.norm2(x + self.drop2(ca))

        
        ff = self.ffn(x)
        x = self.norm3(x + self.drop3(ff))

        return x  # (B, T, D)



class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x



class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        max_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.d_model = d_model

    def forward(self, tgt, enc_out, tgt_mask=None, cross_mask=None):
        
        x = self.tok_emb(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, cross_mask)
        return x



class Transformer(nn.Module):
   

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_enc_layers: int = 4,
        n_dec_layers: int = 4,
        d_ff: int = 1024,
        max_enc_len: int = 512,
        max_dec_len: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            vocab_size, d_model, n_heads, d_ff,
            n_enc_layers, max_enc_len, dropout, pad_idx,
        )
        self.decoder = Decoder(
            vocab_size, d_model, n_heads, d_ff,
            n_dec_layers, max_dec_len, dropout, pad_idx,
        )
        
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

   
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def make_src_mask(self, src):
        
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        
        B, T = tgt.size()

        
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)

        
        causal = torch.tril(
            torch.ones(T, T, device=tgt.device, dtype=torch.bool)
        ).unsqueeze(0).unsqueeze(0)

        
        return pad_mask & causal

    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_out, tgt_mask, cross_mask):
        return self.decoder(tgt, enc_out, tgt_mask, cross_mask)

    def project(self, dec_out):
        return self.out_proj(dec_out)

    
    def forward(self, src, tgt):
        
        src_mask   = self.make_src_mask(src)          
        tgt_mask   = self.make_tgt_mask(tgt)          
        cross_mask = self.make_src_mask(src)           

        enc_out = self.encode(src, src_mask)           
        dec_out = self.decode(tgt, enc_out,
                              tgt_mask, cross_mask)    
        logits  = self.project(dec_out)                
        return logits



def count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable