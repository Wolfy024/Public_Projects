from torch import nn

class MNIST(nn.Module):
    def __init__(self, input_channel, feature_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, feature_dim, kernel_size=3, stride=1, padding=1),  # Output: [32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: [32, 14, 14]
            nn.Conv2d(feature_dim, feature_dim*2, kernel_size=3, stride=1, padding=1),  # Output: [64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: [64, 7, 7]
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = self.fc(x)
        return x
