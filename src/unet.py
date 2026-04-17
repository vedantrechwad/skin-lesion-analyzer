import torch
import torch.nn as nn
from torchvision import transforms

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class SimpleUNet(nn.Module):
    """
    A standard Deep Learning U-Net for Lesion Boundary Segmentation.
    Expects input shape: (B, 3, 256, 256)
    Outputs raw logits shape: (B, 1, 256, 256)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Decoder
        self.up_trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up1 = DoubleConv(512, 256)
        
        self.up_trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = DoubleConv(256, 128)
        
        self.up_trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # Encoder passes
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        
        # Decoder passes
        u1 = self.up_trans1(d4)
        u1 = self.up1(torch.cat([u1, d3], dim=1))
        
        u2 = self.up_trans2(u1)
        u2 = self.up2(torch.cat([u2, d2], dim=1))
        
        u3 = self.up_trans3(u2)
        u3 = self.up3(torch.cat([u3, d1], dim=1))
        
        return self.outc(u3)

def load_unet_model(checkpoint_path, device):
    """
    Load the trained U-Net model from Colab.
    """
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

class TinyBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Task2TinyUNet(nn.Module):
    """
    A lightweight U-Net optimized for 5-channel attribute segmentation.
    Outputs: [B, 5, H, W] for the 5 attributes.
    """
    def __init__(self):
        super().__init__()
        self.d1 = TinyBlock(3, 32)
        self.d2 = TinyBlock(32, 64)
        self.d3 = TinyBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c1 = TinyBlock(128, 64)
        
        self.u2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c2 = TinyBlock(64, 32)
        
        self.out = nn.Conv2d(32, 5, kernel_size=1)
        
    def forward(self, x):
        d1 = self.d1(x)
        p1 = self.pool(d1)
        
        d2 = self.d2(p1)
        p2 = self.pool(d2)
        
        d3 = self.d3(p2)
        
        u1 = self.u1(d3)
        c1 = self.c1(torch.cat([u1, d2], dim=1))
        
        u2 = self.u2(c1)
        c2 = self.c2(torch.cat([u2, d1], dim=1))
        
        return self.out(c2)

def load_attribute_unet(checkpoint_path, device):
    """
    Load the trained Task 2 Attribute U-Net.
    """
    model = Task2TinyUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

