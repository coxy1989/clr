import math
from torch import nn, optim, Tensor

def _triangular_f(it:int, ss:int, min_lr:float, max_lr:float):
    'TODO: docstring'
    # calculate number of completed cycles
    cyc = math.floor(it / (ss * 2))
    # calculate number of completed iterations in current cycle
    it_cyc = it - (cyc * 2 * ss)
    # calculate distance from lr_max iteration
    mid_dist = math.fabs(it_cyc - ss)
    # scale lr difference
    scalar = mid_dist / ss
    return min_lr + (1 - scalar) * (max_lr - min_lr)

def _triangular2_f(it:int, ss:int, min_lr:float, max_lr:float):
    'TODO: docstring'
    # calculate number of completed cycles
    cyc = math.floor(it / (ss * 2))
    # calculate number of completed iterations in current cycle
    it_cyc = it - (cyc * 2 * ss)
    # calculate distance from lr_max iteration
    mid_dist = math.fabs(it_cyc - ss)
    # scale lr difference
    scalar = mid_dist / ss
    return min_lr + (1 - scalar) * ((max_lr - min_lr) / 2 ** cyc)

def _decay_f(it:int, ss:int, min_lr:float, max_lr:float):
    'TODO: docstring'
    if it > ss:
        return min_lr
    diff = max_lr - min_lr
    scalar = it / ss
    return max_lr - (scalar * diff)

def _exp_f(it:int, gamma:float, initial_lr:float):
    'TODO: docstring'
    return initial_lr * gamma ** it

def _exp_range_f(it:int, gamma:float, ss:int, min_lr:float, max_lr:float):
    'TODO: docstring'
    return  _triangular_f(it, ss, min_lr * (gamma ** it), max_lr * (gamma ** it))

class FixedScheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, optimizer:optim.Optimizer):
        'TODO: docstring'
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        # _LRScheduler increments `last_epoch` on each call to `step()`
        return [pg['lr'] for pg in self.optimizer.param_groups]

class TriangularScheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, step_size:int, min_lr:float, max_lr:float, optimizer:optim.Optimizer):
        'TODO: docstring'
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        it = self.last_epoch
        lr = _triangular_f(it, self.step_size, self.min_lr, self.max_lr)
        return [lr] * len(self.optimizer.param_groups)

class Triangular2Scheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, step_size:int, min_lr:float, max_lr:float, optimizer:optim.Optimizer):
        'TODO: docstring'
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        it = self.last_epoch
        lr = _triangular2_f(it, self.step_size, self.min_lr, self.max_lr)
        return [lr] * len(self.optimizer.param_groups)

class ExpRangeScheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, gamma:float, step_size:int, min_lr:float, max_lr:float, optimizer:optim.Optimizer):
        'TODO: docstring'
        self.gamma = gamma
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        it = self.last_epoch
        lr = _exp_range_f(it, self.gamma, self.step_size, self.min_lr, self.max_lr)
        return [lr] * len(self.optimizer.param_groups)

class DecayScheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, step_size:int, min_lr:float, max_lr:float, optimizer:optim.Optimizer):
        'TODO: docstring'
        self.step_size = step_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        it = self.last_epoch
        lr = _decay_f(it, self.step_size, self.min_lr, self.max_lr)
        return [lr] * len(self.optimizer.param_groups)

class ExpScheduler(optim.lr_scheduler._LRScheduler):
    'TODO: docstring'

    def __init__(self, gamma:float, initial_lr:float, optimizer:optim.Optimizer):
        'TODO: docstring'
        self.gamma = gamma
        self.initial_lr = initial_lr
        super().__init__(optimizer)

    def get_lr(self):
        'TODO: docstring'
        it = self.last_epoch
        lr = _exp_f(it, self.gamma, self.initial_lr)
        return [lr] * len(self.optimizer.param_groups)
