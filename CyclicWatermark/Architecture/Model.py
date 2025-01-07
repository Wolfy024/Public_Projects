import torch
import torch.nn as nn
import torch.nn.functional as F


# Spectral Normalization Wrapper
def spectral_norm(module):
    return nn.utils.spectral_norm(module)


# Convolutional Down-sampling Block with Residual Connection and Batch Normalization
class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# Convolutional Up-sampling Block with Batch Normalization
class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.cat([x, residual], dim=1)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# U-Net Generator with Residual Blocks and Batch Normalization
class UnetGAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetGAN, self).__init__()
        self.conv1 = ConvDownBlock(in_channels, 64)
        self.conv2 = ConvDownBlock(64, 128)
        self.conv3 = ConvDownBlock(128, 256)
        self.conv4 = ConvDownBlock(256, 512)
        self.conv5 = ConvDownBlock(512, 1024)
        self.bottom = ConvDownBlock(1024, 2048)
        self.up1 = ConvUpBlock(2048, 1024)
        self.up2 = ConvUpBlock(1024, 512)
        self.up3 = ConvUpBlock(512, 256)
        self.up4 = ConvUpBlock(256, 128)
        self.up5 = ConvUpBlock(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.mp(x1))
        x3 = self.conv3(self.mp(x2))
        x4 = self.conv4(self.mp(x3))
        x5 = self.conv5(self.mp(x4))
        x = self.bottom(self.mp(x5))
        x = self.up1(x, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.final_conv(x)
        return x


# Discriminator with Spectral Normalization and Leaky ReLU
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1))
        self.fc1 = spectral_norm(nn.Linear(256 * 32 * 32, 1))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x


# Example usage
if __name__ == '__main__':
    generator = UnetGAN(in_channels=6, out_channels=3)
    discriminator = Discriminator()
    x = torch.randn(1, 6, 128, 128)
    generated_image = generator(x)
    prediction = discriminator(generated_image)
    print(f'Generated image size: {generated_image.size()}')
    print(f'Discriminator prediction size: {prediction.size()}')
