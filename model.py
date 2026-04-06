"""
model.py
--------
Multi-task classification model for the asbestosis chest X-ray pipeline.

Public API
----------
MultiTaskModel        : Shared-backbone model with one binary head per task.
_build_multitask_model: Factory that dispatches to ViT or CNN builders.
_split_head_backbone_params: Separate head and backbone parameters for
                             differential learning rates.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    vit_b_16,
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """
    Shared-backbone multi-task classification model.

    A single backbone (CNN or ViT) extracts image features that are shared
    across all tasks.  Each task gets its own independent linear head
    (optionally preceded by dropout) that outputs a single logit for
    ``BCEWithLogitsLoss``.

    Args:
        backbone:         Feature extractor module (final head already removed).
        head_in_features: Dimensionality of the backbone output vector.
        task_names:       Ordered list of task names.
        head_dropout:     Dropout probability before each task head.  Pass
                          ``0`` or ``0.0`` to disable.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_in_features: int,
        task_names: List[str],
        head_dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_names)
        heads = {}
        for name in self.task_names:
            if head_dropout and head_dropout > 0:
                heads[name] = nn.Sequential(nn.Dropout(p=head_dropout), nn.Linear(head_in_features, 1))
            else:
                heads[name] = nn.Linear(head_in_features, 1)
        self.heads = nn.ModuleDict(heads)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, torch.Tensor) and feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return {name: head(feats).view(-1) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# ViT builder
# ---------------------------------------------------------------------------

def _build_vit_multitask(
    task_names: List[str],
    no_pretrained: bool,
    head_dropout: float,
) -> MultiTaskModel:
    """
    Build a ViT-B/16 multi-task model adapted for 1-channel (grayscale) input.

    The RGB patch-projection weights are averaged across the channel dimension
    to initialise the grayscale conv.

    Args:
        task_names:    List of task names for which to create output heads.
        no_pretrained: If ``True``, use random initialisation.
        head_dropout:  Dropout probability before each task head.

    Returns:
        :class:`MultiTaskModel` with a ViT-B/16 backbone.
    """
    weights = None if no_pretrained else ViT_B_16_Weights.DEFAULT
    try:
        base = vit_b_16(weights=weights)
    except Exception as e:
        print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
        base = vit_b_16(weights=None)

    old = base.conv_proj
    new = torch.nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    )
    with torch.no_grad():
        if old.weight.shape[1] == 3:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        else:
            new.weight.copy_(old.weight)
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
    base.conv_proj = new

    head_in_features = base.hidden_dim
    base.heads = nn.Identity()
    return MultiTaskModel(base, head_in_features=head_in_features, task_names=task_names, head_dropout=head_dropout)


# ---------------------------------------------------------------------------
# CNN builders
# ---------------------------------------------------------------------------

def _replace_first_conv_in_channels(module: nn.Module, in_channels: int = 1) -> bool:
    """
    Replace the first ``nn.Conv2d`` in a module tree to accept ``in_channels``.

    Copies weights by averaging across the original input channels where
    possible.

    Args:
        module:      Root module to search.
        in_channels: Desired number of input channels (typically 1 for grayscale).

    Returns:
        ``True`` if a conv was replaced, ``False`` otherwise.
    """
    def _try_replace(parent: nn.Module, name: str, conv: nn.Conv2d) -> bool:
        if conv.in_channels == in_channels:
            return True
        new = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
        )
        with torch.no_grad():
            if conv.weight.ndim == 4 and conv.weight.shape[1] >= 1 and new.weight.shape[1] == 1:
                new.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
            if conv.bias is not None and new.bias is not None:
                new.bias.copy_(conv.bias)
        setattr(parent, name, new)
        return True

    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            return _try_replace(module, name, child)
        if _replace_first_conv_in_channels(child, in_channels=in_channels):
            return True
    return False


# Registry for CNN backbones:
# name → (factory_fn, weights_class, fallback_feature_dim, head_attribute_name)
_CNN_CONFIGS: Dict[str, tuple] = {
    "resnet18":           (resnet18,           ResNet18_Weights,           512,  "fc"),
    "efficientnet_b0":    (efficientnet_b0,    EfficientNet_B0_Weights,    1280, "classifier"),
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights, 576,  "classifier"),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights, 960,  "classifier"),
}


def _build_cnn_backbone(model_name: str, no_pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Build a CNN backbone for multi-task X-ray classification.

    Looks up the model in ``_CNN_CONFIGS``, loads (optional) pretrained weights,
    reads the head's ``in_features``, replaces the head with ``nn.Identity()``,
    and converts the first Conv2d to accept 1-channel (grayscale) input by
    averaging the pretrained RGB weights.

    Args:
        model_name:    One of the keys in ``_CNN_CONFIGS``.
        no_pretrained: If ``True``, initialise with random weights.

    Returns:
        ``(backbone_module, head_in_features)``

    Raises:
        SystemExit: If ``model_name`` is not in the registry.
    """
    name = str(model_name).strip().lower()
    if name not in _CNN_CONFIGS:
        valid = "vit_b_16, " + ", ".join(_CNN_CONFIGS)
        raise SystemExit(f"Unknown model '{model_name}'. Choose from {valid}.")

    factory, weights_cls, fallback_features, head_attr = _CNN_CONFIGS[name]
    weights = None if no_pretrained else weights_cls.DEFAULT
    try:
        base = factory(weights=weights)
    except Exception as e:
        print(f"Warning: failed to load pretrained weights ({e}); falling back to random init.")
        base = factory(weights=None)

    head = getattr(base, head_attr)
    try:
        last = head[-1] if isinstance(head, nn.Sequential) else head
        in_features = int(last.in_features)
    except Exception:
        in_features = fallback_features

    setattr(base, head_attr, nn.Identity())
    _replace_first_conv_in_channels(base, in_channels=1)
    return base, in_features


def _build_multitask_model(
    model_name: str,
    task_names: List[str],
    no_pretrained: bool,
    head_dropout: float,
) -> MultiTaskModel:
    """
    Build a :class:`MultiTaskModel` for the given backbone architecture.

    Dispatches to :func:`_build_vit_multitask` for ``"vit_b_16"`` and to
    :func:`_build_cnn_backbone` for all other architectures.

    Args:
        model_name:    Backbone name (see ``_CNN_CONFIGS`` or ``"vit_b_16"``).
        task_names:    List of task names for which heads are created.
        no_pretrained: If ``True``, use random backbone initialisation.
        head_dropout:  Dropout probability before each task head.

    Returns:
        Initialised :class:`MultiTaskModel`.
    """
    name = str(model_name).strip().lower()
    if name == "vit_b_16":
        return _build_vit_multitask(task_names, no_pretrained=no_pretrained, head_dropout=head_dropout)
    backbone, head_in_features = _build_cnn_backbone(name, no_pretrained=no_pretrained)
    return MultiTaskModel(backbone, head_in_features=head_in_features, task_names=task_names, head_dropout=head_dropout)


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _split_head_backbone_params(model: MultiTaskModel):
    """
    Split model parameters into head and backbone groups.

    Useful for applying a lower learning rate to the backbone while keeping
    the task heads at the base learning rate.

    Returns:
        ``(head_params, backbone_params)`` – lists of ``nn.Parameter``.
    """
    head_ids = {id(p) for p in model.heads.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return head_params, backbone_params
