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

        # Initial convolution layer (same as before)
        self.conv1 = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU()

        # Define the ResNet blocks
        self.layer1 = ResidualBlock(num_features, num_features * 2)
        self.layer2 = ResidualBlock(num_features * 2, num_features * 4)
        self.layer3 = ResidualBlock(num_features * 4, num_features * 8)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_features * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial conv layer
        x = self.layer1(x)  # First residual block
        x = self.layer2(x)  # Second residual block
        x = self.layer3(x)  # Third residual block
        x = self.gap(x).view(x.size(0), -1)  # Global Average Pooling
        x = self.fc(x)  # Fully connected layers
        return x
