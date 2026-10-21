from typing import Any

import torch
import torch.nn as nn
from torch.nn import MaxPool2d


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor = 2,mode = "bilinear", align_corners= True
        )

        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self,x,skip):
        x = self.up(x)

        x = torch.cat((x,skip), dim=1)

        x = self.conv(x)

        return x

model = Up(512,256,256)
x = torch.randn(1,512,32,16)
skip = torch.randn(1,256,64,32)
y = model(x,skip)
print(y.shape)