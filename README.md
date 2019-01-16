# CLR

[![CircleCI](https://circleci.com/gh/coxy1989/clr.svg?style=svg)](https://circleci.com/gh/coxy1989/clr)

This repository provides an implementation of the *Learning Rate Range Test* and *Cyclical Learning Rates (CLR)* as originally described in the paper: *Cyclical Learning Rates for Training Neural Networks* by Leslie N. Smith [1].

What's in the box?

- [Implementations](https://github.com/coxy1989/clr/blob/master/modules/schedulers.py) of the *triangular*, *triangular2*, *decay* and *exp_range* policies.
- An [implementation](https://github.com/coxy1989/clr/blob/master/modules/schedulers.py#L58) of the *Learning Rate Range Test* described in section 3.3.
- [Ports](https://github.com/coxy1989/clr/blob/master/modules/model.py) of the *'full'* and *'quick'* CIFAR10 Caffe models to pytorch.
- [Experiments](https://nbviewer.jupyter.org/github/coxy1989/clr/blob/master/notebooks/headline_experiment.ipynb) which verify the efficacy of *CLR* combined with the *Learning Rate Range Test* in reducing training time, compared to the caffe default CIFAR10 configuration.

## Quickstart

The experiments performed in this repository were conducted on a paperspace instance with a Nvidia Quadro P4000 GPU.

1. `git clone git@github.com:coxy1989/clr.git`

2. `cd clr` 

3. `conda env create -f environment.yml`

3. `source activate clr`

4. `jupyter notebook`

### Run the Notebooks

[Experiments](https://nbviewer.jupyter.org/github/coxy1989/clr/blob/master/notebooks/headline_experiment.ipynb) - reproduce results from the *Result* section of this README.

[Figures](https://nbviewer.jupyter.org/github/coxy1989/clr/blob/master/notebooks/headline_figures.ipynb) - render figures from the *Result* section of this README.

[Schedulers](https://nbviewer.jupyter.org/github/coxy1989/clr/blob/master/notebooks/schedulers.ipynb) - render graphs for learning rate policies.

## Results

### LR Policies

*Fixed Policy:*

|  LR 	| Start 		| End 	|
|---	|---			|---	|
|  0.001	| 0 		| 60,000|
|  0.0001	| 60,000 	| 65,000|
|  0.00001| 65,000 	| 70,000|

*CLR Policy (triangular):*

|  Step Size | Min LR	| Max LR 	| Start| End |
|---			|---		|---		|---	|---|
|  2000| 0.0025 | 0.01| 0 | 35,000|

*Fixed vs CLR Training Result (average of 5 training runs):*

|  LR Policy | Iterations | Accuracy (%)|
|---|---|---|
|  *fixed* | 70,000  | 76.0 |
|  *CLR (triangular policy)* | **20,000**  | 76.0 |

- The CLR policy achieves the same accuracy in `20,000` iterations as that obtained by the fixed policy in `70,000` iterations:

![figure_1](./images/run.png)

*Learning Rate Range Test:*

- Suitable boundries for the CLR policy are at `~0.0025`, where the accuracy starts to increase and at `~0.01`, where the *Learning Rate Range Test* plot becomes ragged:

![figure_1](./images/lrrt.png)


The architecture with which the experiments above were conducted was [ported](https://github.com/coxy1989/clr/blob/master/modules/model.py) from caffe's CIFAR10 ['quick train test'](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt) configuration.

## References

[1] Leslie N. Smith. Cyclical Learning Rates for Training Neural Networks. [arXiv:1506.01186](https://arxiv.org/pdf/1506.01186.pdf), 2015.
