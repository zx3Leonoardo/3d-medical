import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=20, training=True):
        super(Unet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)

        self.decoder4 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder1 = nn.Conv3d(32, 20, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(20, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1,1,1), mode='trilinear'),
            nn.Softmax(dim=1)
        )
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4,4,4), mode='trilinear'),
            nn.Softmax(dim=1)
        )
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8,8,8), mode='trilinear'),
            nn.Softmax(dim=1)
        )
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16,16,16), mode='trilinear'),
            nn.Softmax(dim=1)
        )
    
    def forward(self,x):
        x = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = x
        x = F.relu(F.max_pool3d(self.encoder2(x), 2, 2))
        t2 = x
        x = F.relu(F.max_pool3d(self.encoder3(x), 2, 2))
        t3 = x
        x = F.relu(F.max_pool3d(self.encoder4(x), 2, 2))

        output1 = self.map1(x)
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2,2,2), mode='trilinear'))
        x = torch.add(x, t3)
        output2 = self.map2(x)
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2,2,2), mode='trilinear'))
        x = torch.add(x, t2)
        output3 = self.map3(x)
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2,2,2), mode='trilinear'))
        x = torch.add(x, t1)

        x = F.relu(F.interpolate(self.decoder1(x), scale_factor=(2,2,2), mode='trilinear'))
        output4 = self.map4(x)

        if self.training:
            return output1, output2, output3, output4
        else:
            return output4

        