import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


from time import perf_counter

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from Transformer.train import Batch, LabelSmoothing, NoamOpt, run_epoch
from Transformer.transformer import MyTransformer

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
        src = torch.tensor(data, requires_grad=False)
        tgt = torch.tensor(data, requires_grad=False)
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


if __name__ == '__main__':
    vocab = 11
    criterion = LabelSmoothing(size=vocab, padding_idx=0, smoothing=0.0)
    model = MyTransformer(vocab, vocab, N=2)
    model_opt = NoamOpt(model.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if torch.cuda.is_available():
        model.cuda()
        print('using GPU!')

    for epoch in range(100):
        model.train()
        run_epoch(data_gen(vocab, 10, 30, 20), model, SimpleLossCompute(model.model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(vocab, 10, 30, 5), model, SimpleLossCompute(model.model.generator, criterion, None)))
