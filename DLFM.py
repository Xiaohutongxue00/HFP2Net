import torch
import torch.nn as nn
from einops import rearrange

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

class ECAAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        """
        :param channel: 输入特征的通道数
        :param gamma: 控制自适应卷积核大小的参数
        :param b: 平移项
        """
        super(ECAAttention, self).__init__()
        # 自适应选择卷积核大小
        kernel_size = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size // 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化，得到 (B, C, 1, 1)
        y = self.avg_pool(x)

        # 转换为 (B, C)
        y = y.squeeze(-1).transpose(-1, -2)

        # 通过1D卷积建模通道之间的交互
        y = self.conv(y)

        # 激活并恢复为 (B, C, 1, 1) 形状
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)

        # 输出加权特征
        return x * y.expand_as(x)


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x


def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = torch.einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


class RelPosEmb(nn.Module):
    def __init__(
            self,
            block_size,
            rel_size,
            dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


class FixedPosEmb(nn.Module):
    def __init__(self, window_size, overlap_window_size):
        super().__init__()
        self.window_size = window_size
        self.overlap_window_size = overlap_window_size

        attention_mask_table = torch.zeros((window_size + overlap_window_size - 1),
                                           (window_size + overlap_window_size - 1))
        attention_mask_table[0::2, :] = float('-inf')
        attention_mask_table[:, 0::2] = float('-inf')
        attention_mask_table = attention_mask_table.view(
            (window_size + overlap_window_size - 1) * (window_size + overlap_window_size - 1))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten_1 = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords_h = torch.arange(self.overlap_window_size)
        coords_w = torch.arange(self.overlap_window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten_2 = torch.flatten(coords, 1)

        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.overlap_window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.overlap_window_size - 1
        relative_coords[:, :, 0] *= self.window_size + self.overlap_window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.attention_mask = nn.Parameter(attention_mask_table[relative_position_index.view(-1)].view(
            1, self.window_size ** 2, self.overlap_window_size ** 2
        ), requires_grad=False)

    def forward(self):
        return self.attention_mask


class MWSAttention(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(MWSAttention, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head ** -0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head=self.dim_head
        )
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c=self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        # split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head=self.num_spatial_heads),
                         (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn += self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head=self.num_spatial_heads,
                        h=h // self.window_size, w=w // self.window_size, p1=self.window_size, p2=self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qk_ratio=4, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        # 使用1x1卷积替代线性层
        self.q = nn.Conv2d(dim, self.qk_dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, self.qk_dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        # 投影层也改为卷积
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成Q [B, qk_dim, H, W]
        q = self.q(x)
        # 重组为多头 [B, num_heads, H*W, head_dim]
        q = q.reshape(B, self.num_heads, self.qk_dim // self.num_heads, H * W).permute(0, 1, 3, 2)

        # 空间降采样处理
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            _, _, H_sr, W_sr = x_.shape
            x_ = x_.reshape(B, C, H_sr * W_sr).permute(0, 2, 1)
            # 生成K/V [B, sr_h*sr_w, qk_dim/dim]
            k = self.k(x_).permute(0, 2, 1).reshape(B, self.qk_dim, H_sr, W_sr)
            v = self.v(x_).permute(0, 2, 1).reshape(B, C, H_sr, W_sr)
        else:
            k = self.k(x)
            v = self.v(x)

        k = k.reshape(B, self.num_heads, self.qk_dim // self.num_heads, H * W).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class BiFusion_block(nn.Module):
    def __init__(self, channel, ch_out, drop_rate=0.1):
        super(BiFusion_block, self).__init__()
        self.eca1 = ECAAttention(channel)
        self.eca2 = ECAAttention(channel)
        self.W = Conv(channel * 2, ch_out, 3, bn=True, relu=True)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        self.spatial_attn = MWSAttention(channel, 8, 0.5, 2, 16, False)

    def forward(self, g, x):
        g = self.eca1(g)
        x = self.eca2(x)
        fuse = self.W(torch.cat([g, x], 1))

        fuse = self.spatial_attn(fuse)
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

