import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.GroupNorm(2, out_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(out_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.GroupNorm(2, out_dim)
    )

class unSymUnet(nn.Module):
    def __init__(self, in_chnnel=1, out_channel=20, training=True):
        super().__init__()
        self.out_channel = out_channel
        self.training = training

        self.encoder1 = nn.Conv3d(in_chnnel, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(64, 128, 3, stride=1, padding=1)

        self.decoder4 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(32, 32, 3, stride=1, padding=1)
        self.decoder1 = nn.Conv3d(32, 32, 3, stride=1, padding=1)

        self.gp1 = nn.GroupNorm(8, 128)
        self.gp2 = nn.GroupNorm(8, 64)
        self.gp3 = nn.GroupNorm(8, 32)

        self.up1 = conv_block(128,64)
        self.up2 = conv_block(64, 32)
        self.up3 = conv_block(48, 32)

        self.map = nn.Sequential(
            nn.Conv3d(32, self.out_channel, kernel_size=1, stride=1),
        )
    
    def forward(self, x):
        size_t0 = x.size()
        x = self.gp4(F.relu(F.max_pool3d(self.encoder1(x), 2, 2)))
        size_t1 = x.size()
        t1 = x
        x = self.gp3(F.relu(F.max_pool3d(self.encoder2(x), 2, 2)))
        size_t2 = x.size()
        t2 = x
        x = self.gp2(F.relu(F.max_pool3d(self.encoder3(x), 2, 2)))
        size_t3 = x.size()
        t3 = x
        x = self.gp1(F.relu(F.max_pool3d(self.encoder4(x), 2, 2)))

        x = self.gp2(F.relu(F.interpolate(self.decoder4(x), size=size_t3[2:], mode='trilinear'))) 
        x = self.up1(torch.cat([x, t3], 1))
        x = self.gp3(F.relu(F.interpolate(self.decoder3(x), size=size_t2[2:], mode='trilinear'))) 
        x = self.up2(torch.cat([x, t2], 1))
        x = self.gp3(F.relu(F.interpolate(self.decoder2(x), size=size_t1[2:], mode='trilinear'))) 
        x = self.up3(torch.cat([x, t1], 1))
        x = self.gp3(F.relu(F.interpolate(self.decoder1(x), size=size_t0[2:], mode='trilinear')))
        output = self.map(x)

        return output
