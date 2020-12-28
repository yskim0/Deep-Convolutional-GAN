import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def train_data_loader(batch_size=64, workers=1, shuffle=True):
    """ return training, test dataloader
    Args:
        batch_size : (int) dataloader batchsize
        workers : (int) # of subprocesses
        shuffle : (bool) data shuffle at every epoch
    Returns:
        train_data_loader : torch dataloader obj.
    """

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # maps output to [-1, 1] value range
    ])

    train_dataset = datasets.MNIST(root='./data/', train=True, download=False, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers
    )

    return train_data_loader
