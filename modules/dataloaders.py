import torch
from torchvision import datasets, transforms
from typing import Callable

def channel_avg(train_dl):
    r_sum = g_sum = b_sum = 0
    for x, y in iter(train_dl):
        r_sum += x[:,0,:,:].numpy().ravel().sum()
        b_sum += x[:,1,:,:].numpy().ravel().sum()
        g_sum += x[:,2,:,:].numpy().ravel().sum()
    num_pix = (50000 * 32 * 32)
    r_ave = r_sum / num_pix
    b_ave = b_sum / num_pix
    g_ave = g_sum / num_pix
    rgb_ave = [r_ave, b_ave, g_ave]
    return rgb_ave

def get_cifar10_data_loaders(data_dir:str,
        batch_size:int,
        train_transform:Callable=transforms.ToTensor(),
        test_transform:Callable=transforms.ToTensor(),
        shuffle:bool=True,
        num_workers:int=4,
        pin_memory:bool=True):
    'TODO: docstring'
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return (train_loader, test_loader)
