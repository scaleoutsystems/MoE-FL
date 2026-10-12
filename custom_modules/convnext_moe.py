from torchvision.models.convnext import CNBlock
from torchvision.models import convnext_tiny 
from .convnext_moe_blocks import MoEConvNeXtWrapper, MoECNBlock
    
#For convnext_tiny    
def make_last_block_moe_factory(num_experts=4, top_k=1, mlp_ratio=4, capacity_ratio=1.0,bpr=True,context_aware=False,allowed_experts=None):
    # convnext_tiny stage dims and number of blocks
    stage_depth = {96: 3, 192: 3, 384: 9, 768: 3}

    # which stages should get MoE
    moe_dims = {384, 768}  # last 2 stages

    # Blocks created for each dim
    seen = {d: 0 for d in stage_depth}

    def block(dim, layer_scale, stochastic_depth_prob, norm_layer=None):
        # increment count for this stage dim
        if dim not in seen:
            raise ValueError(f"Unexpected dim={dim}. If not convnext_tiny, update stage_depth.")
        seen[dim] += 1

        is_last_block_in_stage = (seen[dim] == stage_depth[dim])
        use_moe = (dim in moe_dims) and is_last_block_in_stage

        if use_moe:
            return MoECNBlock(
                dim=dim,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                layer_scale=layer_scale,
                norm_layer=norm_layer,
                capacity_ratio=capacity_ratio,
                context_aware=context_aware,
                bpr=bpr,
                allowed_experts=allowed_experts,
            )
        else:
            return CNBlock(
                dim=dim,
                layer_scale=layer_scale,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=norm_layer,
            )

    return block

# Model factory 
def convnext_moe_model_fn(num_experts=4,top_k=1,mlp_ratio=2,capacity_ratio=1.0,
                          collect_expert_stats=False,num_classes=1000,bpr=True,context_aware=False,allowed_experts=None):
    block_factory = make_last_block_moe_factory(
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
        bpr=bpr,
        context_aware=context_aware,
        allowed_experts=allowed_experts,
    )
    return MoEConvNeXtWrapper(convnext_tiny(weights=None, num_classes=num_classes, block=block_factory),collect_expert_stats)