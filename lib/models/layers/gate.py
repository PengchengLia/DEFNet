import torch
import torch.nn as nn

class Gate(nn.Module):
    def __init__(self, dim=768, patch_size=16, num_heads=12, N=64):
        super().__init__()
        # self.patch_size = patch_size
        # self.num_heads = num_heads
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Conv2d(dim*2,dim*4,1,bias=True)
        self.fc1 = nn.Linear(N,N*2,bias=True)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        # self.fc2 = nn.Conv2d(dim*4,dim*2,1,bias=True)
        self.fc2 = nn.Linear(N*2,320,bias=True)
        self.act2 = nn.Sigmoid()
        # self.act2 = nn.ReLU(inplace=True)
    def forward(self, x):
        # B, N, C = x.shape
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x