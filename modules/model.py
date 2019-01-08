from torch import nn

class LRN(nn.Module):
    '''Implementation of local response normalisation. The pytorch implementation does not afford
    within channel normalisation - https://github.com/pytorch/pytorch/issues/653'''

    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1), 
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0)) 
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class Cifar10Net_quick(nn.Module):
    'Port of caffe architecture: https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_train_test.prototxt'

    def __init__(self, num_classes=10):
        'TODO: docstring'
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ip1 = nn.Linear(64 * 3 * 3, 64)
        self.ip2 = nn.Linear(64, 10)

    def forward(self, x):
        'TODO: docstring'
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ip1(out.view(out.shape[0], out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.ip2(out)
        return out

class Cifar10Net_full(nn.Module):
    'Port of caffe architecture: https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_full_train_test.prototxt'

    def __init__(self, num_classes=10):
        'TODO: docstring'
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU()
        self.norm1 = LRN(local_size=3, alpha=5e-5, beta=0.75)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.norm2 = LRN(local_size=3, alpha=5e-5, beta=0.75)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)
        self.ip1 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        'TODO: docstring'
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.norm2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.ip1(out.view(out.shape[0], out.shape[1] * out.shape[2] * out.shape[3]))
        return out

