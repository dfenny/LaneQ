import torch
import torch.nn as nn
import torch.nn.functional as F

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer, similar to the YOLO models."""
    def __init__(self, in_channels, out_channels, pool_size=5):
        super(SPPF, self).__init__()
        
        mid_channels = in_channels // 2  # Reduce channels first
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)  # Adding batch normalization
        self.conv2 = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Adding Batch normalization

        # Multiple pooling sizes (This is apparently better than our previous approach)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2) # 5x5 pooling
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 3x3 pooling
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)  # No pooling

    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))  # Step 1: Reduce channels
        x2 = self.pool1(x1)  # Step 2: Large pooling (5x5)
        x3 = self.pool2(x1)  # Step 3: Medium pooling (3x3)
        x4 = self.pool3(x1)  # Step 4: No pooling

        x = torch.cat((x1, x2, x3, x4), dim=1)  # Step 5: Concatenate across channels
        x = torch.relu(self.bn2(self.conv2(x)))  # Step 6: Reduce channels again
        return x

class CNN_SPPF(nn.Module):
    def __init__(self, in_channels=3, out_dim=1):
        super(CNN_SPPF, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # Adding batch normalization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # Adding batch normalization
        
        # SPPF layer and adaptive pooling
        self.sppf = SPPF(32, 64, pool_size=5)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 16)  # Output size matches feature map channels
        self.fc2 = nn.Linear(16, out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.sppf(x)  
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1)  # Flattening
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x # Output the regression value