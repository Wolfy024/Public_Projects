import torch
from torch import nn


class FashionMNIST1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FashionMNIST1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14 * 14 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = FashionMNIST1(1, 10)
    x = torch.randn(64, 1, 28, 28)
    print(model(x).shape)
