from functools import partial

import torch
import torch.nn as nn
import math
from torch.nn import functional as F

# from util.pos_embed import get_2d_sincos_pos_embed
from timm.models.layers import DropPath, Mlp, PatchEmbed

from torch.cuda.amp import autocast


class FourierAttention(nn.Module):
    def __init__(
        self,
        attn_drop=0.,
    ) -> None:
        super().__init__()
        self.attn_drop = nn.Dropout(attn_drop, in_place=False)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        with autocast(enabled=False):
            att = torch.fft.fft(torch.fft.fft(x.float(), dim=-1), dim=-2).real
        
        att = self.attn_drop(att) 
        return att


class Block(nn.Module):
    def __init__(
        self, dim, drop=0., attn_drop=0.,  mlp_ratio=4,
        drop_path=0., norm_layer=nn.LayerNorm
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(attn_drop=attn_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        x = self.norm1(x + self.drop_path1(self.attn(x)))
        x = self.norm2(x + self.drop_path2(self.mlp(x)))
        return x


class VisionFTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool='token',
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        class_token=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        embed_layer=PatchEmbed,
        drop_path_rate=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        block_fn=Block,
    ) -> None:
        super().__init__()

        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
            bias=False,
        )
        num_patches = self.patch_embed.num_patches

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        ) if class_token else None
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches+1, embed_dim) * .02
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
            ) for _ in range(depth)
        ])
        self.fc_norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.class_token:
            cls_tokens = self.class_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        if self.global_pool == 'token':
            x = x[:, 0]
        elif self.global_pool == 'avg':
            x = x[:, 1:, :].mean(dim=1)
        else:
            raise NotImplementedError(f'Global pooling {self.global_pool} is not implemented')
        x = self.fc_norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def vifnet_small_patch16(**kwargs):
    model = VisionFTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        **kwargs
    )
    return model


def vifnet_base_patch16(**kwargs):
    model = VisionFTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        **kwargs
    )
    return model


def vifnet_large_patch16(**kwargs):
    model = VisionFTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        mlp_ratio=4,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        **kwargs
    )
    return model


def vifnet_huge_patch14(**kwargs):
    model = VisionFTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        mlp_ratio=4,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        **kwargs
    )
    return model