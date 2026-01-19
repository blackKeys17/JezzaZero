import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO - Try WDL value head to stabilise training

# I used same ordering of dimensions as the AlphaZero paper, need to swap them here to match channel-first convention of PyTorch
class ResNet(nn.Module):
    def __init__(self, in_channels, res_channels, res_blocks, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution layer, sets right number of channels for residual block
        self.conv_in = nn.Conv2d(in_channels, res_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(res_channels)

        # Residue block tower
        self.res_tower = nn.Sequential(*[ResBlock(res_channels) for _ in range(res_blocks)])

        # Policy head
        self.conv_pol = nn.Conv2d(res_channels, out_channels, 1, 1)

        # Value head - WDL
        self.conv_val = nn.Conv2d(res_channels, 1, 1, 1)
        self.bn_val = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Shuffle dimensions into right order (-1 autofills for batch size)
        x = x.permute(0, 3, 1, 2)

        # Residual body
        x = F.relu(self.bn(self.conv_in(x)))
        x = self.res_tower(x)

        # Policy head (note that this will output raw logits, makes it easier to mask illegal moves before softmaxing)
        policy = self.conv_pol(x)
        # Shuffle dimensions back
        policy = policy.permute(0, 2, 3 ,1)

        # Value head
        value = F.relu(self.bn_val(self.conv_val(x)))
        value = F.relu(self.fc1(torch.flatten(value, start_dim=1)))
        # value = self.dropout(value)

        # Softmax into WDL distribution
        value = self.fc2(value).squeeze(1)

        return policy, value

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)

        # 2 convolution layers
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = F.relu(self.bn(self.conv2(out)))
        
        # Avoid in-place
        out = out + residual
        return out

if __name__ == "__main__":
    net = ResNet(19, 16, 2, 64)
    for i in range(1):
        x = torch.rand([1, 8, 8, 19])
        p, v = net(x)
        print(p.shape)
        print(v.shape)