import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.downsample(identity)
        x = self.relu(x)
        return x


class FashionMNIST2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FashionMNIST2, self).__init__()
        self.res1 = ResidualBlock(in_channels, 64)
        self.res2 = ResidualBlock(64, 128)
        self.res3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 256, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.res3(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = FashionMNIST2(1, 10)
    x = torch.randn(64, 1, 28, 28)
    print(model(x).shape)
