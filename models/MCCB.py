import torch
import torch.nn as nn

class MultiScale_Contextual_Convolution_Block(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        # tmp_ch = out_channels // 2
        # tmp_ch = max(min(in_channels // 2, out_channels // 2), 32)
        if in_channels <= 128:
            tmp_ch = min(in_channels, 64)
        else:
            tmp_ch = min(in_channels // 4, 128)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, tmp_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(tmp_ch),
            nn.ReLU(inplace=True)
        )
        # self.local_conv = SingleConvLayer(tmp_ch, tmp_ch // 2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
        #                                   bias=True)
        self.conv1 = SingleConvLayer(tmp_ch, tmp_ch, 3, 1, 5, 5, 1, True)
        self.conv2 = SingleConvLayer(tmp_ch, tmp_ch, 3, 1, 3, 3, 1, True)
        self.conv3 = SingleConvLayer(tmp_ch, tmp_ch, 3, 1, 1, 1, 1, True)

        self.global_conv = nn.Sequential(
            nn.Conv2d(tmp_ch * 3, tmp_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(tmp_ch),
            nn.ReLU(inplace=True)
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(tmp_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(tmp_ch, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.sa = convolutional_Attention_Module()

    def forward(self, x):
        f = self.conv1x1(x)
        # feature1 = self.local_conv(f)
        feature1 = self.conv1(f)
        f1 = feature1 + f
        feature2 = self.conv2(f1)
        f2 = feature2 + f
        feature3 = self.conv3(f2)
        join_feature = torch.cat((feature1, feature2, feature3), dim=1)

        x = self.global_conv(join_feature)
        x = x + f
        # x = self.conv1x1_2(x)
        sa_x = self.sa(x)
        if self.residual:
            x = self.bn_relu(sa_x + x)
        else:
            x = sa_x
        outputs = self.conv4(x)
        return outputs

class SingleConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(SingleConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class convolutional_Attention_Module(nn.Module):
    def __init__(self, kernel_size=7):
        super(convolutional_Attention_Module, self).__init__()
        padding = kernel_size // 2
        self.conv = Conv(2, 1, 3, bn=True, relu=True, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x * inputs
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
