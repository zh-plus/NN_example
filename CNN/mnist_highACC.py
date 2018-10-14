import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from tools import *

'''
    Gets to 99.60% test accuracy after 30 epochs.
    (Maybe enhanced after 50 epochs, but not apparent.)
    93 seconds per epoch on a Nvidia 1060 6G GPU.
    Don't use only CPU to run it, which would cost 3.3 hours per epoch using some GPU.
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # drop_prob = 0.3
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5,
                      stride=1, padding=2),  # (64,28,28)
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(kernel_size=2),  # (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),  # (128,28,28)
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # (128,14,14)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2),  # (256,14,14)
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            # nn.MaxPool2d(2),  # (32,3,3)
            # nn.Dropout(drop_prob)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),  # (256,14,14)
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2)  # (256,7,7)
            # nn.Dropout(drop_prob)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),  # (512, 7, 7)
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),  # (1024, 7, 7)
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2),  # (1024,3,3)
        )

        self.FC1 = nn.Sequential(
            nn.Linear(1024 * 3 * 3, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048)
        )
        self.FC2 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print(x.size())
        # input('test')
        x = x.view(x.size(0), -1)
        x = self.FC1(x)
        x = self.FC2(x)
        output = self.out(x)
        return output


def train(model, device, train_loader, optimizer, epoch, loss_rec):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_rec.append(loss)  # record current loss
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()), end=' === ')
            print('with learning rate:', optimizer.param_groups[0]['lr'])


def test(model, device, test_loader, acc_rec):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    acc_rec.append(acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))


def main():
    torch.manual_seed(1)

    EPOCH = 30
    BATCH_SIZE = 128
    learning_rate = 0.01002
    # DOWNLOAD_MNIST = not Path('./mnistData').exists()
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/MNIST-data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/MNIST-data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs)

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-08, momentum=0.7)

    # for group in optimizer.param_groups:
    #     group['initial_lr'] = learning_rate
    # lr_decay = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1, last_epoch=20)

    loss_rec, acc_rec = [], []
    for epoch in range(1, EPOCH + 1):
        # lr_decay.step(epoch)
        tic = time.time()
        train(model, device, train_loader, optimizer, epoch, loss_rec)
        toc = time.time()
        print(toc - tic, 's')
        test(model, device, test_loader, acc_rec)
        # torch.save(model.state_dict(), 'C:\\Users\\10578\\Desktop\\MNIST\\model' + str(epoch) + '.pth')

    print('=============')
    print(max(acc_rec))
    # get_error_visible(model, device, test_loader)
    plot(loss_rec, acc_rec)


if __name__ == '__main__':
    main()
