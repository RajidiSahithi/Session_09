import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools


def print_samples(loader, count=16):
    """Print samples input images
    
    Args:
        loader (DataLoader): dataloader for training data
        count (int, optional): Number of samples to print. Defaults to 16.
    """
    # Print Random Samples
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if not count % 8 == 0:
        return
    fig = plt.figure(figsize=(15, 5))
    for imgs, labels in loader:
        for i in range(count):
            ax = fig.add_subplot(int(count/8), 8, i + 1, xticks=[], yticks=[])
            ax.set_title(f'{classes[labels[i]]}')
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))
        break

def print_rand_train(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    plt.imshow(torchvision.utils.make_grid(images[:10]))
    