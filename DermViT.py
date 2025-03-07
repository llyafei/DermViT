import torch
import torch.nn as nn
from timm.models import register_model
from typing import Tuple, Union
from timm.models.layers import trunc_normal_, LayerNorm2d
from fairscale.nn.checkpoint import checkpoint_wrapper
from DermViT_block import DermViT_block, BasicLayer
from collections import OrderedDict
from thop import profile, clever_format
from prettytable import PrettyTable
from modules import MutiScalePyramidTransformer


class DermViT(nn.Module):
    def __init__(self, depth=None, in_chans=3, num_classes=7, embed_dim=None,
                 head_dim=32, representation_size=None,
                 drop_path_rate=None,
                 drop_path_rate_CGLU=None,
                 mlp_ratios=None,
                 norm_layer=nn.BatchNorm2d,
                 pre_head_norm_layer=None,
                 n_wins=None,
                 topks=None,
                 side_dwconv: int = 5
                 ):
        super().__init__()
        if topks is None:
            topks = [1, 4, 16, -2]
        if n_wins is None:
            n_wins = [7, 7, 7, 7]
        if embed_dim is None:
            embed_dim = [96, 192, 384, 768]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if depth is None:
            depth = [2, 2, 6, 2]
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        ############ downsample layers (patch embeddings) ######################

        self.downsample_layers = nn.ModuleList()
        # patch embedding: conv-norm-MPT
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0], kernel_size=(4, 4), stride=(4, 4)),
            norm_layer(embed_dim[0]),
            MutiScalePyramidTransformer(in_dim=embed_dim[0], num_heads=1, mlp_ratio=1.0, attn_drop=0., drop_path=0., drop=0.),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            # patch merging: norm-conv
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(embed_dim[i]),
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(2, 2), stride=(2, 2)),
                MutiScalePyramidTransformer(in_dim=embed_dim[i + 1], num_heads=1, mlp_ratio=1.0, attn_drop=0., drop_path=0.,
                                   drop=0.)
            )
            self.downsample_layers.append(downsample_layer)

        ##########################################################################

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in embed_dim]

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        for i in range(4):
            stage = BasicLayer(
                dim=embed_dim[i],
                depth=depth[i],
                num_heads=nheads[i],
                mlp_ratio=mlp_ratios[i],
                drop_path=dp_rates[sum(depth[:i]):sum(depth[:i + 1])],
                drop_path_rate_CGLU=drop_path_rate_CGLU,
                n_win=n_wins[i],
                topk=topks[i],
                side_dwconv=side_dwconv,
            )
            self.stages.append(stage)

        ##########################################################################

        pre_head_norm = pre_head_norm_layer or norm_layer
        self.norm = pre_head_norm(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim[-1], representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], self.num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def he_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=torch.sqrt(torch.tensor(2.0 / m.weight.size(1))))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_layers(self):
        layers = nn.ModuleList()
        for i in range(4):
            layer = nn.Sequential(
                self.downsample_layers[i],
                self.stages[i]
            )
            layers.append(layer)
        return layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        for i in range(4):
            # [N,C,H,W] -> [N,C,H,W]
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            # [N,C,H,W] -> [N,C,H,W]
            x = self.stages[i](x)
        x = self.norm(x)
        N, C, H, W = x.shape
        # [N,C,H,W] -> [N,HW,C]
        x = x.flatten(2).transpose(1, 2)
        x = self.pre_logits(x)
        x = x.transpose(1, 2).reshape(N, -1, H, W)
        return x

    def forward(self, x):
        # [N,C,H,W]
        x = self.forward_features(x)
        # flatten: -> [N,C,HW]
        # mean(-1): -> [N,C]
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x

    def forward1(self, x):
        outputs = []
        for i in range(4):
            for name, module in self.downsample_layers[i].named_children():
                x = module(x)
                if name in ["0", "2"]:
                    outputs.append(x)
            # output.append(x)
            x = self.stages[i](x)
            # outputs.append(x)
        x = self.norm(x)
        N, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pre_logits(x)
        x = x.transpose(1, 2).reshape(N, -1, H, W)
        return outputs

@register_model
def DermViT_base(pretrained=False, num_classes=7, drop_path_rate=0., drop_path_rate_CGLU=0.,
                 **kwargs):
    model = DermViT(depth=[2, 2, 6, 2],
                    num_classes=num_classes,
                    embed_dim=[96, 192, 384, 768],
                    mlp_ratios=[4, 4, 4, 4],
                    head_dim=32,
                    norm_layer=nn.BatchNorm2d,
                    n_wins=(7, 7, 7, 7),
                    topks=(1, 4, 16, -1),
                    side_dwconv=5,
                    drop_path_rate=drop_path_rate,
                    drop_path_rate_CGLU=drop_path_rate_CGLU,
                    representation_size=768,
                    **kwargs)
    if pretrained:
        checkpoint = torch.load("pretrain_weights/DermViT_base_best.pth")
        print("load biformer_stl pretrained weights from {}".format("pretrain_weights/DermViT_base_best.pth"))
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del checkpoint["model"][k]
        print(model.load_state_dict(checkpoint["model"], strict=False))
    return model


if __name__ == '__main__':
    model = DermViT_base(num_classes=7)
    x = torch.rand(1, 3, 224, 224)
    macs, params = profile(model, inputs=(x,))
    macs, flops, params = clever_format([macs, 2 * macs, params], "%.3f")
    table = PrettyTable()
    table.field_names = ["MACs", "FLOPs", "params"]
    table.add_row([macs, flops, params])
    print(table)
    print(model)
