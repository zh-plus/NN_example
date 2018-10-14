from time import time

import torch
import matplotlib.pyplot as plt


def get_error_visible(model, device, test_loader):
    model.eval()
    torch.no_grad()
    errors = []

    tic = time.time()
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        label = output.max(1)[1]

        error = [t for t in zip(data.cpu(), label.cpu(), target) if t[1] != t[2]]
        errors.extend(error)

    toc = time.time()
    print(toc - tic, 's')

    for img, error_lable, correct_lable in errors:
        img = img.squeeze().numpy()
        plt.imshow(img, cmap='gray')
        plt.xlabel('correct: ' + str(correct_lable.item()))
        plt.ylabel('predict: ' + str(error_lable.item()) + '     ', rotation=0)
        plt.show()


def plot(loss_rec, acc_rec):
    """
    Plot the loss and accuracy of model
    :param loss_rec: int[]
    :param acc_rec: int[]
    """
    plt.subplot(2, 1, 1)
    plt.plot(loss_rec)
    plt.ylabel('loss')
    # plt.xlabel('per ' + str(BATCH_SIZE / 100) + ' batch')

    plt.subplot(2, 1, 2)
    plt.plot(acc_rec, 'o-')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.show()