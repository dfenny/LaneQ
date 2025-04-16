"""
unet.py

Implementation of the UNet architecture for image segmentation, following the original paper.
Reference: https://paperswithcode.com/method/u-net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d => BatchNorm2d => ReLU) * 2.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of intermediate channels. If None, defaults to out_channels.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Apply the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W).
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block with max pooling followed by double convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """
        Apply downscaling block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H/2, W/2).
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: upsample (bilinear or transpose convolution) then double convolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bilinear : bool, optional
        If True, use bilinear upsampling; otherwise, use ConvTranspose2d. Default is True.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Apply upscaling block by concatenating encoder features with upsampled decoder features.

        Parameters
        ----------
        x1 : torch.Tensor
            Decoder input tensor of shape (N, in_channels, H, W).
        x2 : torch.Tensor
            Corresponding encoder feature map tensor of shape (N, out_channels, H*2, W*2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H*2, W*2).
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    1x1 convolution to map to the desired number of output channels.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Apply 1x1 convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_channels, H, W).
        """
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet model for image segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    out_channels : int, optional
        Number of output segmentation channels. Default is 1.
    bilinear : bool, optional
        Whether to use bilinear upsampling. Default is True.

    Attributes
    ----------
    inc : DoubleConv
        Initial convolution block.
    down1 : Down
        First downscaling block.
    down2 : Down
        Second downscaling block.
    down3 : Down
        Third downscaling block.
    down4 : Down
        Bottleneck downscaling block.
    up1 : Up
        First upscaling block.
    up2 : Up
        Second upscaling block.
    up3 : Up
        Third upscaling block.
    up4 : Up
        Fourth upscaling block.
    outc : OutConv
        Final 1x1 convolution block.
    """

    def __init__(self, in_channels, out_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor) # This is the bottleneck layer
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        """
        Forward pass of the UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Output logits tensor of shape (N, out_channels, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        # return torch.sigmoid(logits) # Since we are doing binary segmentation