import torch
from torch import nn
import torch.nn.functional as F

class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1x1 = CBR(in_c * 2, out_c, kernel_size=1, padding=0)
        self.conv3x3 = CBR(out_c, out_c, act=False)
        # self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear')
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1x1(x)

        s1 = x
        x = self.conv3x3(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.conv3x3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class output_block(nn.Module):

    def __init__(self, in_c, out_c=1):
        super().__init__()

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.fuse=CBR(in_c*3,in_c, kernel_size=3, padding=1)
        self.c1 = CBR(in_c, in_c, kernel_size=3, padding=1)
        # self.c2 = CBR(128, 64, kernel_size=1, padding=0)
        self.c2 = CBR(in_c, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(32, out_c, kernel_size=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):

        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        x3 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear')

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x)

        x = self.up_2x2(x)
        x = self.c1(x)
        x = self.up_2x2(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.sig(x)
        return x

