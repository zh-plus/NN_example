import matplotlib.pyplot as plt

import numpy as np
import torch


def im_show(img, std=0.5, mean=0.5, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # De-Normalization
    img = img / mean + std
    np_img = img.numpy()
    if one_channel:
        plt.imshow(np_img, cmap='Greys')
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)))


def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
