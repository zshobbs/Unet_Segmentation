import torch
import torch.nn as nn


def twoconv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Max pool for down sampeling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder side of network
        self.encode_1 = twoconv(3, 64)
        self.encode_2 = twoconv(64, 128)
        self.encode_3 = twoconv(128, 256)
        self.encode_4 = twoconv(256, 512)
        self.encode_5 = twoconv(512, 1024)

        # decode side of network
        self.exspand_1 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2)
        
        self.decode_1 = twoconv(1024, 512)
        
        self.exspand_2 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2)
        
        self.decode_2 = twoconv(512, 256)
        
        self.exspand_3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2)
        
        self.decode_3 = twoconv(256, 128)
        
        self.exspand_4 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2)
        
        self.decode_4 = twoconv(128, 64)
        
        self.out = nn.Conv2d(64, 5, kernel_size=1)
    
    def forward(self, image):
        # Encoder
        x1 = self.encode_1(image)
        x = self.max_pool(x1)

        x2 = self.encode_2(x)
        x = self.max_pool(x2)

        x3 = self.encode_3(x)
        x = self.max_pool(x3)

        x4 = self.encode_4(x)
        x = self.max_pool(x4)
        
        x5 = self.encode_5(x)

        
        # decoder
        x = self.exspand_1(x5)
        x = torch.cat((x, x4), axis=1)
        x = self.decode_1(x)
        
        x = self.exspand_2(x)
        x = torch.cat((x, x3), axis=1)
        x = self.decode_2(x)
        
        x = self.exspand_3(x)
        x = torch.cat((x, x2), axis=1)
        x = self.decode_3(x)
        
        x = self.exspand_4(x)
        x = torch.cat((x, x1), axis=1)
        x = self.decode_4(x)
        x = self.out(x)
        return x

ian = torch.rand(1, 3, 256,256)
t = UNet()
t(ian)
