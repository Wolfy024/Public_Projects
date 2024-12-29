import torch
import torch.nn as nn


# Define a Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output channels are different, adjust input size using a 1x1 convolution
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the input (skip connection)
        out = self.relu(out)  # ReLU activation after adding the residual
        return out


# CIFAR10 model with ResNet-like residual blocks
class CIFAR10Model(nn.Module):
    def __init__(self, input_channels, num_features, num_classes):
        super(CIFAR10Model, self).__init__()
        self.conv1 = ResidualBlock(input_channels, num_features)
        self.conv2 = ResidualBlock(num_features, num_features * 2)
        self.conv3 = ResidualBlock(num_features * 2, num_features * 4)
        self.conv4 = ResidualBlock(num_features * 4, num_features * 8)
        self.conv5 = ResidualBlock(num_features * 8, num_features * 16)
        self.pool = nn.MaxPool2d(4)
        self.fc = nn.Linear(num_features * 16 * 8 * 8, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x