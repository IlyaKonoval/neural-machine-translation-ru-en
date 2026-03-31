import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size

        self.q_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        q = self.q_linear(query).view(batch_size, query_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(key).view(batch_size, key_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(value).view(batch_size, value_len, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, query_len, self.embed_size)
        return self.fc_out(out)


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(forward))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_size, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, heads)
        self.cross_attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attention = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention))
        cross_attention = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attention))
        forward = self.feed_forward(x)
        x = self.norm3(x + self.dropout(forward))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, ff_hidden_size, dropout, max_len=5000):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.word_embedding(x) * (self.embed_size ** 0.5)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_size, num_layers, heads, ff_hidden_size, dropout, max_len=5000):
        super().__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, heads, ff_hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.word_embedding(x) * (self.embed_size ** 0.5)
        x = self.position_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.fc_out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size=256,
        num_layers=4,
        heads=8,
        ff_hidden_size=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, ff_hidden_size, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_layers, heads, ff_hidden_size, dropout)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_sub_mask = tgt_sub_mask.expand(batch_size, 1, tgt_len, tgt_len)
        return tgt_pad_mask & tgt_sub_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
