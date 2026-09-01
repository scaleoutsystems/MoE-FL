import torch.nn as nn
from typing import Iterable
import timm
from .convnext_moe_blocks import MoEConvNeXtWrapper, MoECNBlock

def inject_moe_into_convnext(
    model: nn.Module,
    moe_stages: Iterable[int] = (2, 3),   # last two stages
    num_experts: int = 4,
    top_k: int = 1,
    mlp_ratio: int = 4,
    capacity_ratio: float = 1.0,
    blocks_per_stage: str = "last",       # "last" or "all"
):
    """Swap selected timm ConvNeXtBlocks for MoECNBlock, in place."""
    moe_stages = set(moe_stages)
    for s_idx, stage in enumerate(model.stages):
        if s_idx not in moe_stages:
            continue
        blocks = stage.blocks
        dim = blocks[0].conv_dw.in_channels
        targets = range(len(blocks)) if blocks_per_stage == "all" else [len(blocks) - 1]
        for b_idx in targets:
            blocks[b_idx] = MoECNBlock(
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                capacity_ratio=capacity_ratio,
                layer_scale=0.0,
            )
    return model


def convnext_atto_moe_model_fn(
    num_experts: int = 4,
    top_k: int = 1,
    mlp_ratio: int = 4,
    capacity_ratio: float = 1.0,
    moe_stages: Iterable[int] = (2, 3),
    blocks_per_stage: str = "last",
    collect_expert_stats: bool = False,
    num_classes: int = 100,
    pretrained: bool = False,
):
    base = timm.create_model("convnext_atto", pretrained=pretrained, num_classes=num_classes)
    base = inject_moe_into_convnext(
        base,
        moe_stages=moe_stages,
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
        blocks_per_stage=blocks_per_stage,
    )
    return MoEConvNeXtWrapper(base, collect_expert_stats)