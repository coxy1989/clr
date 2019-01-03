import torch
from torchvision import datasets, transforms
from typing import Callable

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
