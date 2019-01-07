from torch import nn

def init_weights(module:nn.Module):
    if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
        nn.init.normal_(module.weight.data, std=0.01, mean=0)
        nn.init.constant_(module.bias.data, 0)

class Cifar10Net(nn.Module):
    'TODO: docstring'

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
        self.ip1 = nn.Linear(64 * 7 * 7, 64)
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
        out = self.ip1(out.view(out.shape[0], out.shape[1] * out.shape[2] * out.shape[3]))
        out = self.ip2(out)
        return out

