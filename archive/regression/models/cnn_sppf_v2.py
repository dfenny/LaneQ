import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SPPF_v2(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer, similar to the YOLO models."""

    def __init__(self, in_channels, out_channels, pool_sizes=[1, 3, 5]):
        super().__init__()

        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels * (len(pool_sizes)+1), out_channels, kernel_size=1)

        # Define multiple max-pooling layers for different scales (kernel sizes)
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        x1 = self.conv1(x)  # Step 1: Reduce channels

        pooled_features = [x1]  # List to store pooled outputs

        # Apply pooling with different kernel sizes
        for pool in self.pools:
            x1 = pool(x1)
            pooled_features.append(x1)  # Apply each pooling operation to the feature map

        # Concatenate the pooled feature maps across the channel dimension
        x = torch.cat(pooled_features, dim=1)  # Step 5: Concatenate across channels

        # Reduce channels again
        x = self.conv2(x)  # Step 6: Apply the second 1x1 convolution

        return x


class CNN_SPPF_v2(nn.Module):

    def __init__(self, in_channels=3, out_dim=3, pool_sizes=[1, 3, 5]):

        super().__init__()

        # Convolutional layers with varying kernel sizes and Batch Normalization
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(128)

        # SPPF layer and adaptive pooling (To ensure input to FC is static)
        pool_sizes = [3, 3, 3]
        self.sppf = SPPF_v2(64, 128, pool_sizes=pool_sizes)  # Using 3 different pooling sizes (5, 9, 13)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Independent of size of image
        self.fc2 = nn.Linear(64, out_dim)
        

    def forward(self, x):

        # First Convolutional Block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        # Second Convolutional Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)

        # Third Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.max_pool(x)

        # fourth conv block
        # x = F.relu(self.bn4(self.conv4(x)))

        x = self.sppf(x)  # Apply SPPF
        x = self.global_pool(x)  # Shape here should be (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x  # For classification, not activation since loss function takes care of it
