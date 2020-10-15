import os

from time import perf_counter

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from Transformer.train import Batch, LabelSmoothing, NoamOpt, run_epoch, test
from Transformer.transformer import MyTransformer, subsequent_mask

torch.cuda.set_device(7)

"""
We can begin by trying out a simple copy-task.
Given a random set of input symbols from a small vocabulary, 
the goal is to generate back those same symbols.
"""


def data_gen(vocab, d_sentence, batch_size, n_batch):
    """Generate random data for a src-tgt copy test"""
    for i in range(n_batch):
        data = torch.from_numpy(np.random.randint(1, vocab, size=(batch_size, d_sentence)))
        data[:, 0] = 1
        src = data.clone().detach()
        tgt = data.clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    if torch.cuda.is_available():
        src, src_mask, ys = src.cuda(), src_mask.cuda(), ys.cuda()

    memory = model.model.encode(src, src_mask)
    for i in range(max_len - 1):
        out = model.model.decode(memory, src_mask,
                           ys.clone().detach(),
                           subsequent_mask(ys.size(1)).clone().detach().type_as(src.data))
        prob = model.model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == '__main__':
    vocab = 11
    criterion = LabelSmoothing(size=vocab, padding_idx=0, smoothing=0.0)
    my_model = MyTransformer(vocab, vocab, N=2)
    model_opt = NoamOpt(my_model.d_model, 1, 400, torch.optim.Adam(my_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if torch.cuda.is_available():
        my_model.cuda()
        print('using GPU!')

    for epoch in range(100):
        print(f'Epoch: {epoch}')
        my_model.train()
        run_epoch(data_gen(vocab, 10, 1000, 20), my_model, SimpleLossCompute(my_model.model.generator, criterion, model_opt))
        # my_model.eval()
        # print(test(data_gen(vocab, 10, 300, 5), my_model))

    my_model.eval()
    src = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).long()
    src_mask = torch.ones(1, 1, 10)
    result = greedy_decode(my_model, src, src_mask, max_len=10, start_symbol=1)
    print(result)