import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import *
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple
from self_similarity import *


class GISSA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.norm = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        self.qkv = nn.Conv2d(dim, dim * 3, 1, groups=num_heads)
        self.qkv2 = nn.Conv2d(dim, dim * 3, 1, groups=head_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)

        x = ((attn @ v).reshape(B, C, H, W) + x).reshape(B, self.num_heads, self.head_dim, H, W).transpose(1, 2).reshape(B, C, H, W)
        y = self.norm(x)
        x = self.relu(y)
        qkv2 = self.qkv2(x).reshape(B, 3, self.head_dim, self.num_heads, H * W).transpose(0, 1)
        q, k, v = qkv2[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = torch.sign(attn) * torch.sqrt(torch.abs(attn) + 1e-5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, self.head_dim, self.num_heads, H, W).transpose(1, 2).reshape(B, C, H, W) + y
        return x


class SIT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = GISSA(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        cur = self.norm1(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        cur = cur.permute(0, 2, 1).reshape(B, -1, H, W)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)
        x = x.flatten(2).permute(0, 2, 1)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x


class SWSA(nn.Module):
    def __init__(self, channels, calc_attn=True):
        super(SWSA, self).__init__()
        self.channels = channels
        self.calc_attn = calc_attn
        self.scale = channels ** -0.5

        if self.calc_attn:
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels*2, kernel_size=1),
                nn.BatchNorm2d(self.channels*2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1),
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns = None):

        b,c,h,w = x.shape
        x = self.project_inp(x)
        if prev_atns is None:
            q, v = rearrange(
                x, 'b (qv c) h w -> qv b (h w)  c',
                qv=2
            )
            atn = (q @ q.transpose(-2, -1)) * self.scale
            atn = atn.softmax(dim=-1)
            y = (atn @ v)
            y = rearrange(
                y, 'b (h w)  c-> b c h w',
                h=h, w=w
            )

            y = self.project_out(y)
            return y, atn
        else:
            atn = prev_atns
            v = rearrange(
                x, 'b c h w -> b (h w)  c',
            )
            y = (atn @ v) * self.scale
            y = rearrange(
                y, 'b (h w) c-> b c h w',
                h=h, w=w
            )
            y = self.project_out(y)
            return y, prev_atns


class SCA(nn.Module):
    def __init__(self, in_channels, out_channels, shared_depth=1):
        super(SCA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = CBR_2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        modules_gmsa['gmsa_0'] = SWSA(channels=in_channels, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i + 1)] = CBR_2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            modules_gmsa['gmsa_{}'.format(i + 1)] = SWSA(channels=in_channels, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
        return x


class EATN(nn.Module):
    def __init__(self, in_channel, num_classes, dataset):
        super(EATN, self).__init__()
        channels = [128, 64, 192, 256, 32]
        self.in_channel = in_channel
        self.dataset = dataset
        # self.patch_size = 11
        self.conv_channel = CBR_2d(self.in_channel, channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_spatial = CBR_2d(self.in_channel, channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.sit = SIT(channels[0], 16)

        self.sca = SCA(channels[0], channels[0], shared_depth = 1)

        self.ssfe = SSFE(self.in_channel, self.in_channel, self.in_channel, 3)
        self.lamuda = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        if self.dataset == 'HUST2013':
            self.dropout = nn.Dropout(0.)
        elif self.dataset == 'HanChuan':
            self.dropout = nn.Dropout(0.3)
        else:
            self.dropout = nn.Dropout(0.5)
        self.pool = GlobalAvgPool2d()
        self.fc = nn.Linear(channels[0], num_classes, bias=False)

    def forward(self, x):
        B, _, H, W = x.shape
        x_str = self.ssfe(x)
        x_channel = self.conv_channel(x_str)
        x_channel = self.sit(x_channel)
        x_spatial = self.conv_spatial(x_str)
        x_spatial = self.sca(x_spatial)
        lmd = torch.sigmoid(self.lamuda)
        x = lmd * x_channel + (1 - lmd) * x_spatial
        # print((lmd))
        x = self.pool(self.dropout(x)).view(-1, x.shape[1])
        x = self.fc(x)

        return x


if __name__ == '__main__':
    net = model = EATN(200, 16, "IP")
    net.eval()
    print(net)
    input = torch.randn(64, 200, 11, 11)
    y = net(input)
    print(y.shape, count_parameters(net))