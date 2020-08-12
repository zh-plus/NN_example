import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from visualization.utils import im_show, select_n_random
from visualization.model import Net

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
writer = SummaryWriter('runs/fashion_MNIST_experiment_3')


def images_to_prob(net, images):
    output = net(images)

    pred_tensor = torch.argmax(output, 1)
    preds = pred_tensor.numpy().squeeze()

    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_prob(net, images)

    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        im_show(images[idx], one_channel=True)
        ax.set_title(f'{classes[preds[idx]]}, {probs[idx] * 100:.1f}%\n(label: {classes[labels[idx]]})',
                     color=('green' if preds[idx] == labels[idx].item() else 'red'))

    return fig


def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    """

    """
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index], tensorboard_preds, tensorboard_probs, global_step=global_step)
    writer.close()


def train(net, train_loader, optimizer, criterion):
    """

    """
    running_loss = 0.0
    print(len(train_loader))
    for epoch in range(1):
        for i, (inputs, labels,) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 0:
                print(i)
                writer.add_scalar('training loss', running_loss / 1000, epoch * len(train_loader) + i)
                writer.add_figure(f'predictions vs. actual ({i})', plot_classes_preds(net, inputs, labels), global_step=epoch * len(train_loader)
                                                                                                                         + i)
            running_loss = 0.0

    print('Finish Training')


def assess(net, test_loader):
    class_probs, class_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            class_preds_batch = torch.argmax(output, dim=1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    for i in range(len(classes)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)


def main():
    BATCH_SIZE = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, num_workers=0)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('eight_fashion_mnist_images', img_grid)
    writer.add_graph(model, images)

    images, labels = select_n_random(train_set.data, train_set.targets)

    class_labels = [classes[label] for label in labels]

    features = images.view(-1, 28 * 28)

    writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))  # 3 channel (N, C, W, H)

    train(model, train_loader, optimizer, criterion)
    assess(model, test_loader)

    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
