import torch
from torch import nn
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from einops import rearrange

class InteractiveDilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.x_y = DoInteractiveDilatedMDTA(dim, num_heads)
        self.y_x = DoInteractiveDilatedMDTA(dim, num_heads)

    def forward(self, x, y):
        return self.x_y(x, y), self.y_x(y, x)

class DoInteractiveDilatedMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(DoInteractiveDilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, dilation=2, padding=2, groups=dim * 2,
                                   bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q_proj(x))
        kv = self.kv_dwconv(self.kv_proj(y))
        k, v = kv.chunk(2, dim=1)

        # 重排以支持多头计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out

class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        # self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)

        self.cross_attn = InteractiveDilatedMDTA(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        B,D,C = u1.shape
        H, W = int(math.sqrt(D)), int(math.sqrt(D))
        u1 = u1.reshape(B, H, W, C).permute(0, 3, 1, 2) # B,C,H,W; B,H,W,C; B,H*W,C
        u2 = u2.reshape(B, H, W, C).permute(0, 3, 1, 2) #

        v1, v2 = self.cross_attn(u1, u2)
        v1 = v1.permute(0, 2, 3, 1).reshape(B, H*W, C)
        v2 = v2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)

        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2

# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()

        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1 , x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2) # B H*W C
        x2 = x2.flatten(2).transpose(1, 2)

        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)

        merge = self.channel_emb(merge, H, W)

        return merge