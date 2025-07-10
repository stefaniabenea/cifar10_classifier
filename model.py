import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net=nn.Sequential(
            # (B, 3, 32, 32) -> (B, 16, 32, 32)
            nn.Conv2d(in_channels=3, out_channels = 16, kernel_size =3, padding = 1),
            nn.ReLU(),
            #(B, 16, 32, 32) -> (B, 16, 16, 16)
            nn.MaxPool2d(2),
            # (B, 16, 16, 16) -> (B, 32, 16, 16)
            nn.Conv2d(in_channels=16, out_channels = 32, kernel_size =3, padding = 1),
            nn.ReLU(),
            # (B, 32, 16, 16) -> (B, 32, 8, 8)
            nn.MaxPool2d(2),
            # (B, 32, 8, 8) -> (B, 2048)
            nn.Flatten(),
            # (B, 2048) -> (B, 128)
            nn.Linear(in_features=2048, out_features=128),
            nn.ReLU(),
            # (B, 128) -> (B, 10)
            nn.Linear(in_features=128, out_features=10),
                 )
        
    def forward(self,x):
        return self.net(x)
        
    
        
