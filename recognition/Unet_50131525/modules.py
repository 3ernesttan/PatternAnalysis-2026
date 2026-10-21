import torch
import torch.nn as nn


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

class UNet2D(nn.Module):
    def __init__(self,in_channels = 1,num_classes = 4):
        super().__init__()

        #initial convolution
        self.inc = DoubleConv(in_channels, 64)

        #encoder
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)

        #bottleneck
        self.down4 = Down(512,1024)

        #decoder
        self.up1 = Up(1024,512,512)
        self.up2 = Up(512,256,256)
        self.up3 = Up(256,128,128)
        self.up4 = Up(128,64,64)

        #output layer
        self.outc = nn.Conv2d(64,num_classes,kernel_size=1)

    def forward(self,x):
        #encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #decoder
        u1 = self.up1(x5,x4)
        u2 = self.up2(u1,x3)
        u3 = self.up3(u2,x2)
        u4 = self.up4(u3,x1)

        logits = self.outc(u4)
        return logits