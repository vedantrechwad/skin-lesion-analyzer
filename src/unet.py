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
