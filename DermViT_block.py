import torch.nn as nn
import torch
from attention import DermoscopicHierarchicalAttention, Attention, AttentionLePE, DWConv
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from modules import MutiScaleAttention_nhwc, DFG


class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        # B H W C -> B H W C
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        # x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        x = x.view(B, H, W, C)
        return x


class DermViT_block(nn.Module):
    """
        Attention + FFN
    """

    def __init__(self, dim, drop_path=0., num_heads=8, n_win=7, qk_dim=None, qk_scale=None, drop_path_rate_CGLU=0.,
                 topk=4, mlp_ratio=4, side_dwconv=5, layer_scale_init_value=0.3, before_attn_dwconv=3):
        super().__init__()
        qk_dim = qk_dim or dim

        # important to avoid attention collapsing
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        # position embedding
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0

        # attention
        if topk > 0:
            self.attn = DermoscopicHierarchicalAttention(
                dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                qk_scale=qk_scale, topk=topk, side_dwconv=side_dwconv)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == -3:
            self.attn = MutiScaleAttention_nhwc(in_dim=dim, num_heads=num_heads)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.mlp = DFG(dim, drop=drop_path_rate_CGLU)
        # self.adaptor = Adapter(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(torch.tensor(layer_scale_init_value) * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.tensor(layer_scale_init_value) * torch.ones(dim), requires_grad=True)
        else:
            self.use_layer_scale = False

    def forward(self, x):
        """
        x: NCHW tensor

        return: NCHW tensor
        """
        # conv pos embedding
        # (N, C, H, W)
        x = x + self.pos_embed(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # attention & mlp
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))

            res = x
            x = self.norm2(x)
            # (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            # (N, C, H, W) -> (N, C, H, W)
            x = self.mlp(x)
            # (N, C, H, W) -> (N, H, W, C)
            x = x.permute(0, 2, 3, 1)
            x = self.gamma2 * x
            x = res + self.drop_path(x)

            # x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            # (N, H, W, C)
            x = x + self.drop_path(self.attn(self.norm1(x)))
            res = x
            x = self.norm2(x)
            # (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2)
            # (N, C, H, W) -> (N, C, H, W)
            x = self.mlp(x)
            # (N, C, H, W) -> (N, H, W, C)
            x = x.permute(0, 2, 3, 1)
            x = res + self.drop_path(x)

            # (N, H, W, C)
            # x = x + self.drop_path(self.mlp(self.norm2(x)))

        # permute back
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class BasicLayer(nn.Module):
    """
    Stack several Blocks
    """

    def __init__(self, dim, depth, num_heads, n_win, topk, mlp_ratio=4., drop_path=None, side_dwconv=5,
                 drop_path_rate_CGLU=0.):
        super().__init__()
        if drop_path is None:
            drop_path = [0.5, 0.5, 0.5, 0.5]
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([
            DermViT_block(
                dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                drop_path_rate_CGLU=drop_path_rate_CGLU,
                num_heads=num_heads,
                n_win=n_win,
                topk=topk,
                mlp_ratio=int(mlp_ratio),
                side_dwconv=side_dwconv,
                before_attn_dwconv=3,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: NCHW tensor
        Return:
            NCHW tensor
        """
        # block: input[N, C, H, W] -> output [N, C, H, W]
        for blk in self.blocks:
            x = blk(x)
        # x: -> [N, C, H, W]
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"


if __name__ == '__main__':
    # block = BasicLayer(64, 2, 8, 7, 4)
    block = DermViT_block(64)
    input = torch.rand(1, 64, 224, 224)
    output = block(input)
    print(input.size(), output.size())
