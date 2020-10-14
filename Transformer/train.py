import os

from tqdm import tqdm
from time import perf_counter

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from Transformer.transformer import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask"""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.tensor(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    start = perf_counter()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in tqdm(enumerate(data_iter)):
        src, tgt, src_mask, tgt_mask = batch.src, batch.tgt, batch.src_mask, batch.tgt_mask

        if torch.cuda.is_available():
            src, tgt, src_mask, tgt_mask = batch.src.cuda(), batch.tgt.cuda(), batch.src_mask.cuda(), batch.tgt_mask.cuda()

        out = model(src, tgt, src_mask, tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.n_tokens)
        total_loss += loss.item()
        total_tokens += batch.n_tokens.item()
        tokens += batch.n_tokens.item()

        if i % 50 == 1:
            elapsed = perf_counter() - start
            print(f'Epoch Step: {i} Loss: {loss / batch.n_tokens} Tokens per Sec: {elapsed / tokens}')
            start = perf_counter()
            tokens = 0

    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fun(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0

    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)

    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch

    return max(src_elements, tgt_elements)


class NoamOpt:
    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self.lr = 0

    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement lr scheduling described in 'Attention is All You Need'"""
        if not step:
            step = self._step

        rate = self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

        return rate


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def test_opt():
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist, requires_grad=False))


def test_smooth():
    criterion = LabelSmoothing(5, 0, 0.4)
    predict = torch.tensor([[0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0],
                            [0, 0.2, 0.7, 0.1, 0]], dtype=torch.float)
    loss = criterion(predict.log(), torch.tensor([2, 1, 0]).long())

    plt.imshow(criterion.true_dist)
    plt.show()


def test_loss():
    criterion = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.tensor([[0, x / d, 1 / d, 1 / d, 1 / d], ], dtype=torch.float)
        print(predict)
        return criterion(torch.tensor(predict.log()), torch.tensor(torch.tensor([1]).long())).item()

    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()


if __name__ == '__main__':
    test_loss()
