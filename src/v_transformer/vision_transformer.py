"""Vision Transformer backbone based on TimeSformer (Divided Space-Time Attention).

This module implements the image branch used in the solar nowcasting pipeline.
It follows the public TimeSformer implementation while trimming extraneous
features and adapting defaults to 128x128 all-sky image clips with five frames.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Truncated normal initialisation (approximate) matching ViT defaults."""
    # Values more than 2 std devs are redrawn (tail-suppression)
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.mul_(std).add_(mean)
        return tensor


class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * noise


class PatchEmbed(nn.Module):
    """2D patch embedding applied frame-wise with fixed 16x16 patches."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if patch_size != 16:
            raise ValueError("Patch size must be 16 to match paper setup")
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B*T, C, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer block with divided space-time attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_drop_path = DropPath(drop_path)

        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.spatial_drop_path = DropPath(drop_path)

        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.mlp_drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, num_frames: int, num_patches: int) -> Tensor:
        B, total_tokens, dim = x.shape
        cls_token = x[:, :1]
        patch_tokens = x[:, 1:]
        patch_tokens = patch_tokens.reshape(B, num_frames, num_patches, dim)

        # Temporal attention: operate across frames for each spatial location.
        t = self.temporal_norm(patch_tokens)
        t = t.permute(0, 2, 1, 3).reshape(B * num_patches, num_frames, dim)
        t = self.temporal_attn(t)
        t = t.reshape(B, num_patches, num_frames, dim).permute(0, 2, 1, 3)
        patch_tokens = patch_tokens + self.temporal_drop_path(t)

        # Spatial attention: operate within each frame, including cls token.
        s_tokens = self.spatial_norm(patch_tokens)
        cls_repeat = cls_token.expand(B, num_frames, dim)
        cls_repeat = cls_repeat.reshape(B * num_frames, 1, dim)
        s_tokens = s_tokens.reshape(B * num_frames, num_patches, dim)
        s_tokens = torch.cat([cls_repeat, s_tokens], dim=1)
        s_tokens = self.spatial_attn(s_tokens)
        cls_out = s_tokens[:, :1]
        patch_out = s_tokens[:, 1:]

        cls_out = cls_out.reshape(B, num_frames, dim).mean(dim=1, keepdim=True)
        patch_out = patch_out.reshape(B, num_frames, num_patches, dim)

        cls_token = cls_token + self.spatial_drop_path(cls_out)
        patch_tokens = patch_tokens + self.spatial_drop_path(patch_out)

        x = torch.cat([cls_token, patch_tokens.reshape(B, num_frames * num_patches, dim)], dim=1)
        x = x + self.mlp_drop_path(self.mlp(self.norm(x)))
        return x


class TimeSformerEncoder(nn.Module):
    """TimeSformer encoder configured for 5-frame ASI clips with 16x16 patches."""

    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        num_frames: int = 5,
    ) -> None:
        super().__init__()
        if num_frames != 5:
            raise ValueError("Number of frames must be 5 to match paper setup")
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_frames * self.num_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _trunc_normal_(self.cls_token, std=0.02)
        _trunc_normal_(self.pos_embed, std=0.02)
        _trunc_normal_(self.temporal_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor shaped (B, T, C, H, W)
        Returns:
            Tensor shaped (B, embed_dim) representing the fused CLS embedding.
        """
        B, T, C, H, W = x.shape
        if T != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {T}")

        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)
        N = x.shape[1]
        x = x.reshape(B, T, N, self.embed_dim)
        x = x + self.temporal_embed[:, :T].unsqueeze(2)
        x = x.reshape(B, T * N, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, num_frames=T, num_patches=N)

        x = self.norm(x)
        return x[:, 0]


__all__ = ["TimeSformerEncoder"]
