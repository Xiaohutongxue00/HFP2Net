import os
import torch.nn.functional as F
import time

import torch
import torch.nn as nn
from update.CFA import Crosslevel_Aggregation
from update.DLFM import BiFusion_block
from update.GSFM import FeatureFusionModule as FFM
from update.MCCB import MultiScale_Contextual_Convolution_Block
from update.aware_decoder import decoder_block,output_block

from utlis.pvtv2_encoder import pvt_v2_b1
from utlis.pvtv2_encoder import pvt_v2_b2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class HFPNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(HFPNet, self).__init__()

        self.rgb_swin = pvt_v2_b2()
        # self.rgb_swin = smt_t(True)
        self.depth_swin = pvt_v2_b1()

        self.CFA1 = Crosslevel_Aggregation(64, 256)
        self.CFA2 = Crosslevel_Aggregation(128, 512)
        self.CFA3 = Crosslevel_Aggregation(320, 960)
        self.CFA4 = Crosslevel_Aggregation(512, 1344)

        ## -- FFM
        self.fusion1 = BiFusion_block(64, 64)
        self.fusion2 = BiFusion_block(128, 128)


        self.fusion3 = FFM(dim=320, reduction=1, num_heads=4)
        self.fusion4 = FFM(dim=512, reduction=1, num_heads=8)

        self.mccb1 = MultiScale_Contextual_Convolution_Block(64,128,False)
        self.mccb2 = MultiScale_Contextual_Convolution_Block(128,128,False)
        self.mccb3 = MultiScale_Contextual_Convolution_Block(320,128,True)
        self.mccb4 = MultiScale_Contextual_Convolution_Block(512,128,True)

        self.output_high = decoder_block(128, 64)
        self.output_middle = decoder_block(128, 64)
        self.output_low = decoder_block(128, 64)

        self.output_block = output_block (64, 1)


        # self.f_pred = nn.Conv2d(64, 1, kernel_size=1, stride=1)

        self.f_pred = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1,  kernel_size=3,padding=1, bias=True),
            )

    def forward(self, x, d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r1 = (rgb_list[0])
        r2 = (rgb_list[1])
        r3 = (rgb_list[2])
        r4 = (rgb_list[3])

        d1 = (depth_list[0])
        d2 = (depth_list[1])
        d3 = (depth_list[2])
        d4 = (depth_list[3])


        rgb1 = self.CFA1(r1,r2)
        rgb2 = self.CFA2(r1,r2,r3)
        rgb3 = self.CFA3(r2,r3,r4)
        rgb4 = self.CFA4(r4,r3)

        depth1 = self.CFA1(d1,d2)
        depth2 = self.CFA2(d1,d2,d3)
        depth3 = self.CFA3(d2,d3,d4)
        depth4 = self.CFA4(d4,d3)

        fuse4 = self.fusion4(rgb4, depth4)
        fuse3 = self.fusion3(rgb3, depth3)
        fuse2 = self.fusion2(rgb2, depth2)
        fuse1 = self.fusion1(rgb1, depth1)


        fuse_1 = self.mccb1(fuse1)
        fuse_2 = self.mccb2(fuse2)
        fuse_3 = self.mccb3(fuse3)
        fuse_4 = self.mccb4(fuse4)

        f_high = self.output_high(fuse_4,fuse_3)
        f_middle = self.output_middle(fuse_3,fuse_2)
        f_low = self.output_low(fuse_2,fuse_1)

        output = self.output_block(f_low,f_middle,f_high)


        y2 = F.interpolate(self.f_pred(f_high), size=384, mode='bilinear')
        y3 = F.interpolate(self.f_pred(f_middle), size=384, mode='bilinear')
        y4 = F.interpolate(self.f_pred(f_low), size=384, mode='bilinear')
        return output, y2, y3, y4



    def load_pre(self, pre_model_r, pre_model_d):
        pretrained_dict1 = torch.load(pre_model_r)
        pretrained_dict1 = {k: v for k, v in pretrained_dict1.items() if k in self.rgb_swin.state_dict()}
        self.rgb_swin.load_state_dict(pretrained_dict1)
        print(f"RGB PyramidVisionTransformerImpr loading pre_model ${pre_model_r}")

        pretrained_dict = torch.load(pre_model_d)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.depth_swin.state_dict()}
        self.depth_swin.load_state_dict(pretrained_dict)
        print(f"Depth PyramidVisionTransformerImpr loading pre_model ${pre_model_d}")



# class initialConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding='valid', dilation=1):
#         super(initialConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         # self.bn = nn.BatchNorm2d(out_channels)
#         # self.leakyrelu = nn.LeakyReLU()
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


def measure_fps(model, input_tensors, iterations=100, gpu_id=2):
    torch.cuda.empty_cache()  # 清理显存
    model.eval()

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensors = [t.to(device) for t in input_tensors]  # ✅ 逐个移动到 GPU

    with torch.no_grad():
        for _ in range(10):  # 预热
            _ = model(*input_tensors)  # ✅ 传入多个输入

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(*input_tensors)  # ✅ 传入多个输入
    end_time = time.time()

    fps = iterations / (end_time - start_time)
    print(f"FPS: {fps:.2f}")

if __name__ == "__main__":
    import torch
    from thop import profile

    model = HFPNet()

    a = torch.randn(1, 3, 384, 384)
    b = torch.randn(1, 3, 384, 384)
    flops, params = profile(model, (a, b))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
    measure_fps(model, (a, b), gpu_id=2)  # ✅ 传入 (a, b) 作为输入