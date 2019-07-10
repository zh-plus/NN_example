import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings

warnings.filterwarnings('ignore')

plt.ion()


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')


# plt.figure()
# show_landmarks(io.imread(os.path.join('study/data/faces/', image_name)), landmarks)
# plt.show()


class FaceLandMarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(image_name)

        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        sample = self.transform(sample) if self.transform else sample

        return sample


def main():
    face_dataset = FaceLandMarksDataset('study/data/faces/face_landmarks.csv', 'study/data/faces')

    fig = plt.figure()

    for i, sample in enumerate(face_dataset):
        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title(f'Sample #{i}')
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break


class Rescale:
    def __init__(self, outputs_size):
        assert isinstance(outputs_size, (int, tuple)), f'Output size should not be {outputs_size}, but int or tuple!'
        self.output_size = outputs_size

    def __call__(self, sample: dict):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * h / 2
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        new_image = transform.resize(image, (new_h, new_w))

        landmarks *= [new_w / w, new_h / h]

        return {'image': new_image, 'landmarks': landmarks}


class RandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)


if __name__ == '__main__':
    main()
