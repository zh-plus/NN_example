# Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
import copy
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tools import Timer


def clones(module, n):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Bse for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_out = self.encode(src, src_mask)

        return self.decode(encoder_out, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        src_embedding = self.src_embed(src)

        return self.encoder(src_embedding, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt_embedding = self.tgt_embed(tgt)

        return self.decoder(tgt_embedding, memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """

    def __init__(self, d_model, vocal):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocal)

    def forward(self, x):
        x = F.log_softmax(self.proj(x), dim=-1)

        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity, the norm is first as opposed to last.
    """

    def __init__(self, feature_size, dropout_p):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(feature_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size"""
        x = sublayer(self.norm(x))
        x = x + self.dropout(x)

        return x


class Encoder(nn.Module):
    """
    Core encoder is as stack of N layers.
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.feature_size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    """Encoder layer is made up of self-attention and feed forward"""

    def __init__(self, feature_size, self_attn, feed_forward, dropout_p):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(feature_size, dropout_p), 2)
        self.feature_size = feature_size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.feature_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    """
    Decoder is made up of self-attention, src-attention (attention over encoder output),
    and feed forward.
    """

    def __init__(self, feature_size, self_attn, src_attn, feed_forward, dropout_p):
        super(DecoderLayer, self).__init__()
        self.feature_size = feature_size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(feature_size, dropout_p), 3)

    def forward(self, x, memory, src_mask, tat_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tat_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[2](x, self.feed_forward)

        return x


def subsequent_mask(size):
    """Mask out subsequent positions"""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype(int)

    return torch.from_numpy(mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)

    # (N, d_k) @ (d_k, N) -> (N, N)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)

    attn_p = F.softmax(scores, dim=-1)
    if dropout:
        attn_p = dropout(attn_p)

    # (N, N) @ (N, d_v) -> (N, d_v)
    weighted_value = torch.matmul(attn_p, value)

    return weighted_value, attn_p


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0  # d_k == d_v = d_model / h

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears[:3], (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_p=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.linear2 = nn.Linear(d_inner, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_p)

        PE = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        inner_term = position * div_term
        print(inner_term)

        PE[:, 0::2] = torch.sin(inner_term)
        PE[:, 1::2] = torch.cos(inner_term)

        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, x):
        x = x + torch.tensor(self.PE[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)

        return x


def test_pe():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()


# def make_model(src_vocal, tgt_vocab, N=6, d_model=512, d_inner=2048, h=8, dropout_p=0.1):
#     dc = copy.deepcopy
#     attn = MultiHeadAttention(h, d_model)
#     feed_forward = PositionWiseFeedForward(d_model, d_inner, dropout_p)
#     pe = PositionalEncoding(d_model, dropout_p)
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, dc(attn), dc(feed_forward), dropout_p), N),
#         Decoder(DecoderLayer(d_model, dc(attn), dc(attn), dc(feed_forward), dropout_p), N),
#         nn.Sequential(Embeddings(src_vocal, d_model), dc(pe)),
#         nn.Sequential(Embeddings(tgt_vocab, d_model), dc(pe)),
#         Generator(d_model, tgt_vocab)
#     )
#
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform(p)
#
#     return model


class MyTransformer(nn.Module):
    def __init__(self, src_vocal, tgt_vocab, N=6, d_model=512, d_inner=2048, h=8, dropout_p=0.1):
        super(MyTransformer, self).__init__()
        dc = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        feed_forward = PositionWiseFeedForward(d_model, d_inner, dropout_p)
        pe = PositionalEncoding(d_model, dropout_p)

        self.d_model = d_model
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, dc(attn), dc(feed_forward), dropout_p), N),
            Decoder(DecoderLayer(d_model, dc(attn), dc(attn), dc(feed_forward), dropout_p), N),
            nn.Sequential(Embeddings(src_vocal, d_model), dc(pe)),
            nn.Sequential(Embeddings(tgt_vocab, d_model), dc(pe)),
            Generator(d_model, tgt_vocab)
        )

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)


if __name__ == '__main__':
    test_pe()
