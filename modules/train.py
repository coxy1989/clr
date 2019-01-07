import torch
from tqdm.autonotebook import tqdm
from torch import nn, Tensor, optim
from typing import Dict, Callable

def validate(dl:torch.utils.data.DataLoader,
             model:nn.Module,
             criterion:nn.Module,
             device: torch.device):
    'TODO: docstring'
    model.eval()
    total_loss = total_correct =  0
    with torch.no_grad():
        for xs, ys in dl:
            xs = xs.to(device)
            ys = ys.to(device)
            out = model(xs)
            # loss
            loss = criterion(out, ys)
            total_loss += loss.item() * xs.size(0)
            # acc
            correct = (out.max(1)[1] == ys).sum().item()
            total_correct += correct

    return total_loss / len(dl.dataset), total_correct / len(dl.dataset)


def train_batch(xs:Tensor,
                ys:Tensor,
                model:nn.Module,
                criterion:nn.Module,
                optimizer:optim.Optimizer):
    'TODO: docstring'
    model.train()
    optimizer.zero_grad()
    out = model(xs)
    loss = criterion(out, ys)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_run(model:nn.Module,
              train_dl:torch.utils.data.dataloader.DataLoader,
              criterion:nn.Module,
              optimizer:optim.Optimizer,
              scheduler:optim.lr_scheduler._LRScheduler,
              num_it:int,
              on_batch_end:Callable[[int, float, float], None],
              device: torch.device):
    'TODO: docstring'
    iterator = iter(train_dl)
    bar = tqdm(range(num_it))
    for i in bar:
        try:
            xs, ys = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            xs, ys = next(iterator)
        xs = xs.to(device)
        ys = ys.to(device)
        loss = train_batch(xs, ys, model, criterion, optimizer)
        on_batch_end(bar, i, loss, scheduler.get_lr()[0])
        scheduler.step()

def on_batch_end(recorder:Dict,
                 test_dl: torch.utils.data.dataloader.DataLoader,
                 model: nn.Module,
                 criterion: nn.Module,
                 device: torch.device,
                 p_bar:tqdm,
                 it_num:int,
                 trn_loss:float,
                 lr: float):
    'TODO: docstring'
    if it_num == 0 or (it_num + 1) % 500 == 0 :
        recorder['iteration'].append(it_num + 1)
        recorder['trn_loss'].append(trn_loss)
        recorder['lr'].append(lr)
        val_loss, val_acc = validate(test_dl, model, criterion, device)
        recorder['val_loss'].append(val_loss)
        recorder['val_acc'].append(val_acc)
        p_bar.write(f'{trn_loss} | {val_loss} | {val_acc}')
