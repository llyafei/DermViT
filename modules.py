# Copyright (c) Xinzi He
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np


class DFG(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # res = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = x + res
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.psp = PSPModule(sizes=(1, 2, 4, 16), dimension=2)

    def forward(self, x):
        # [B,hw,token_dim]
        B, N, C = x.shape

        # qkv(): -> [B,hw,3 * token_dim]
        # reshape: -> [B,hw,3,num_heads,token_dim_per_head]
        # permute: -> [3,B,num_heads,hw,token_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # split: q,k,v -> [B,num_heads,hw,token_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        v_B, v_head, v_N, v_C = q.shape

        # psp: -> [B,num_heads,new_hw,token_dim_per_head]
        v_pooled = self.psp(v.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)
        # psp: -> [B,num_heads,new_hw,token_dim_per_head]
        k = self.psp(k.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)

        # transpose: -> [B,num_heads,token_dim_per_head,new_hw]
        # @: multiply -> [B,num_heads,hw,new_hw]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply: -> [B,num_heads,hw,token_dim_per_dim]
        # transpose -> [B,hw,num_heads,token_dim_per_dim]
        # reshape -> [B,hw,token_dim]
        x = (attn @ v_pooled).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        # out -> B,H*W,C
        return x


class MutiScaleAttention_nhwc(nn.Module):
    def __init__(self, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.head_dim = in_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.psp = PSPModule(sizes=(1, 2, 4, 16), dimension=2)

    def forward(self, x):
        # input -> [B,H,W,C]
        B, H, W, C = x.shape
        N = H * W
        x = x.view(B, -1, C)
        # qkv(): -> [B,hw,3 * token_dim]
        # reshape: -> [B,hw,3,num_heads,token_dim_per_head]
        # permute: -> [3,B,num_heads,hw,token_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # split: q,k,v -> [B,num_heads,hw,token_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        v_B, v_head, v_N, v_C = q.shape

        # psp: -> [B,num_heads,new_hw,token_dim_per_head]
        v_pooled = self.psp(v.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)
        # psp: -> [B,num_heads,new_hw,token_dim_per_head]
        k = self.psp(k.reshape(v_B * v_head, v_N, v_C)).view(v_B, v_head, -1, v_C)

        # transpose: -> [B,num_heads,token_dim_per_head,new_hw]
        # @: multiply -> [B,num_heads,hw,new_hw]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply: -> [B,num_heads,hw,token_dim_per_dim]
        # transpose -> [B,hw,num_heads,token_dim_per_dim]
        # reshape -> [B,hw,token_dim]
        x = (attn @ v_pooled).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        # out -> B,H,W,C
        x = x.view(B, H, W, C)
        return x


class MutiScalePyramidTransformer(nn.Module):

    def __init__(self, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.attn = Attention(
            in_dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp1 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                        act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        x = x.reshape(B, C, H, W)
        return x


class MutiScalePyramidTransformer_nhwc(nn.Module):

    def __init__(self, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.attn = Attention(
            in_dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp1 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim,
                        act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp1(self.norm2(x)))
        x = x.reshape(B, H, W, C)
        return x


class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 2, 4, 16), dimension=2):
        super(PSPModule, self).__init__()
        # ModuleList
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        # 输入 n,hw,c 输出 n,new_hw,c
        n, hw, c = feats.size()
        # 特征图reshape为 n c h w
        feats = feats.transpose(1, 2).view(n, c, int(np.sqrt(hw)), int(np.sqrt(hw)))
        # list,reshape-> n hw c
        # priors : -> [(n,hw1,c),(n,hw2,c),(n,hw3,c)]
        priors = [stage(feats).view(n, c, -1).transpose(1, 2) for stage in self.stages]
        # cat : -> (n,hw1+hw2+hw3,c)
        center = torch.cat(priors, -2)
        return center


if __name__ == '__main__':
    model = MutiScalePyramidTransformer(in_dim=96, num_heads=1, mlp_ratio=1.0, attn_drop=0.0, drop_path=0, drop=0.)
    x = torch.randn(1, 96, 56, 56)
    out = model(x)
    print(out.shape)
