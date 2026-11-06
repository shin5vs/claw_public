"""U-Net claw segmentation network: timm encoder (ConvNeXt-Tiny or ResNet34) and a skip-connected decoder.

Only this module imports torch, so `clawseg.segmentation` can be imported without it.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )


class ClawUNet(nn.Module):
    """Input: (B, 3, H, W) ImageNet-normalised RGB, H and W multiples of 32. Output: (B, classes, H, W) logits."""

    DECODER = (256, 128, 64, 32)

    def __init__(self, encoder: str = "resnet34", classes: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        channels = self.encoder.feature_info.channels()          # feature levels and their channels
        in_ch = channels[-1]
        blocks = []
        for skip, out in zip(reversed(channels[:-1]), self.DECODER):
            blocks.append(_ConvBlock(in_ch + skip, out))
            in_ch = out
        self.blocks = nn.ModuleList(blocks)
        # If the encoder has no stride-2 level (ConvNeXt starts at stride 4), the full-resolution input image is
        # passed to the last block as a skip, so that the thin interdigital gap is not blurred.
        self.image_skip = self.encoder.feature_info.reduction()[0] > 2
        self.full = _ConvBlock(in_ch + (3 if self.image_skip else 0), 16)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.blocks, reversed(feats[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = block(torch.cat([y, skip], dim=1))
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if self.image_skip:
            y = torch.cat([y, x], dim=1)
        return self.head(self.full(y))
