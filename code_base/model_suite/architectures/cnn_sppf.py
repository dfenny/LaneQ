import torch
import torch.nn as nn
import torch.nn.functional as F

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer, similar to the YOLO models."""
    def __init__(self, in_channels, out_channels, pool_size=5):
        """
        Initialize the SPPF layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        pool_size : int, optional
            Size of the pooling kernel, by default 5.
        """
        super(SPPF, self).__init__()
        
        mid_channels = in_channels // 2
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2)

    def forward(self, x):
        """
        Forward pass of the SPPF layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        x1 = self.conv1(x)  # Step 1: Reduce channels
        x2 = self.pool(x1)  # Step 2: First pooling
        x3 = self.pool(x2)  # Step 3: Second pooling
        x4 = self.pool(x3)  # Step 4: Third pooling
        x = torch.cat((x1, x2, x3, x4), dim=1)  # Step 5: Concatenate across channels
        x = self.conv2(x)   # Step 6: Reduce channels again
        return x

class CNN_SPPF(nn.Module):
    """Convolutional Neural Network with an SPPF layer."""
    def __init__(self, in_channels=3, out_dim=1):
        """
        Initialize the CNN_SPPF model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 3.
        out_dim : int, optional
            Number of output dimensions, by default 1.
        """
        super(CNN_SPPF, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # SPPF layer and adaptive pooling (To ensure input to FC is static)
        self.sppf = SPPF(64, 128, pool_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Independent of size of image
        self.fc2 = nn.Linear(64, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the CNN_SPPF model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_dim).
        """
        x = F.relu(self.conv1(x))  
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))  
        
        x = self.sppf(x)  
        x = self.global_pool(x)  # Shape here should be (batch, 128, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 128)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  

        # return self.sigmoid(x) # Output the regression value
        return x  # For classification, not activation since loss function takes care of it