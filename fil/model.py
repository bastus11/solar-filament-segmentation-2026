"""Segmentation network factory (encoder-decoder from segmentation_models_pytorch + timm encoders)."""
from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def _first_conv(module: nn.Module) -> nn.Conv2d:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise RuntimeError("no conv found")


def build_model(arch: str = "unet", encoder: str = "tu-tf_efficientnetv2_s.in21k_ft_in1k", out_ch: int = 2,
                pretrained: bool = True, decoder_channels=(256, 128, 64, 32, 16), **kw) -> nn.Module:
    """Network with a 3-channel input [image, x/R, y/R].

    The pretrained RGB stem is re-purposed: the image channel receives the sum of the RGB filters
    (== feeding a grey image to the original network) and the two coordinate channels start at zero,
    so the network initially behaves exactly like the pretrained grey-scale model and learns to use
    the position on the solar disk during fine-tuning.
    """
    weights = "imagenet" if pretrained else None
    if arch == "unet":
        net = smp.Unet(encoder, encoder_weights=weights, in_channels=3, classes=out_ch,
                       decoder_channels=decoder_channels, **kw)
    elif arch == "unetpp":
        net = smp.UnetPlusPlus(encoder, encoder_weights=weights, in_channels=3, classes=out_ch,
                               decoder_channels=decoder_channels, **kw)
    elif arch == "upernet":
        net = smp.UPerNet(encoder, encoder_weights=weights, in_channels=3, classes=out_ch, **kw)
    elif arch == "fpn":
        net = smp.FPN(encoder, encoder_weights=weights, in_channels=3, classes=out_ch, **kw)
    elif arch == "segformer":
        net = smp.Segformer(encoder, encoder_weights=weights, in_channels=3, classes=out_ch, **kw)
    else:
        raise ValueError(arch)
    conv = _first_conv(net.encoder)
    if pretrained and conv.in_channels == 3:
        with torch.no_grad():
            w = conv.weight.clone()
            conv.weight[:, 0:1] = w.sum(1, keepdim=True)
            conv.weight[:, 1:] = 0
    return net
