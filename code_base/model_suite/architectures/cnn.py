import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) model.
    """
    def __init__(self, in_channels=3, out_dim=1):
        """
        Initialize the CNN model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default 3.
        out_dim : int, optional
            Number of output dimensions, by default 1.
        """
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # Independent of size of image
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        """
        Forward pass of the CNN model.

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
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  

        return x # Output the regression value