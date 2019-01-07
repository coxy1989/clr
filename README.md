# CLR

This repository provides an implementation of the paper: *Cyclical Learning Rates for Training Neural Networks* by Leslie N. Smith [1]. 

## Contents

- An implementation of the *triangular* and *triangular2* policies specified in section 3.1 and a reproduction of the experiment described in that section and section 4.1.
- An implementation of the *Learning Rate Range Test* described in section 3.3 and a reproduction of the experiment in that section on the CIFAR10 dataset.

## Results

### Table 1

Table of results displaying final accuracies for LR policies on CIFAR10.

|  LR Policy | Iterations | Reported Accuracy (%)| Achieved Accuracy (%)| Diff (%)|
|---|---|---|---|---|
|  *fixed* | 70,000  | 81.4  | 76.0 | 5.4 |
|  *triangular 2* | 25,000  |  81.4 | 74.5 | 6.9 |
|  *decay* |  25,000 | 78.5  | 72.0 |6.5 |
|  *exp* | 70,000  | 79.1  | 68.7  | 10.4 |
|  *exp_range* | 42,000  | 82.2  | 75.7 | 6.5 |

### Figure 1

Graph displaying accuracy vs. iteration for fixed, exponential and CLR policies on CIFAR10.

#### Reported

![figure_1](./images/clr_cifar10.png)

#### Achieved

*TODO*

### Figure 3

Result from a learning rate range test on CIFAR10.

#### Reported

![figure_3](./images/clr_lrrt.png)

#### Achieved

*TODO*

## Notes

## Instructions

## References

[1] Leslie N. Smith. Cyclical Learning Rates for Training Neural Networks. [arXiv:1506.01186](https://arxiv.org/pdf/1506.01186.pdf), 2015.