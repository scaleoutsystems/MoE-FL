import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Optional
from torchvision.ops import StochasticDepth, Permute
from .routers import NoisyTopKRouter
from .patch_dispatcher import PatchDispatcher
from torchvision.models.convnext import CNBlock
from torchvision.models import convnext_tiny 

class MoEConvNeXtWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._aux_modules = [m for m in self.base.modules() if hasattr(m, "aux_loss")]

    def forward(self, x, return_aux: bool = False):
        logits = self.base(x)
        if not return_aux:
            return logits
        return logits, self.get_aux_loss()

    def get_aux_loss(self):
        aux = None
        for m in self._aux_modules:
            a = m.aux_loss
            if a is None:
                continue
            aux = a if aux is None else aux + a
        if aux is None:
            aux = next(self.base.parameters()).new_zeros(())
        return aux

class MoECNBlock(nn.Module, PatchDispatcher): 
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        mlp_ratio: int,
        capacity_ratio : float,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
        )

        self.router = NoisyTopKRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([ConvNeXtExpert(dim, mlp_ratio) for _ in range(num_experts)])
        self.permute_back = Permute([0, 3, 1, 2])
        self.aux_loss = torch.tensor(0.0)
        self.capacity_ratio = capacity_ratio
        self.num_experts = num_experts

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        #self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        x = self.dwconv(input)  # (N, H, W, C)
        N, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # [T, C]

        topi, weights, priority, aux_loss = self.router(x_flat, return_aux=self.training)
        order = torch.argsort(priority, descending=True)
        moe_flat = self.moe_route(x_flat, topi, weights, order)  # [T, C]
        
        if aux_loss is None:
            self.aux_loss = x_flat.new_zeros(())
        else:
            self.aux_loss = aux_loss
            
        #Additional skip connection described in paper       
        x_skip = self.permute_back(x)

        moe_out = moe_flat.view(N, H, W, C)
        moe_out = self.permute_back(moe_out)  # (N, C, H, W)

        moe_out = self.layer_scale * moe_out
        #moe_out = self.stochastic_depth(moe_out)
        
        return input + x_skip + moe_out
    
class ConvNeXtExpert(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        return self.block(x)
    
    
def make_last_block_moe_factory(num_experts=4, top_k=1, mlp_ratio=4, capacity_ratio=1.0):
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
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=norm_layer,
                capacity_ratio=capacity_ratio,
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
def convnext_moe_model_fn(num_experts=4,top_k=1,mlp_ratio=2,capacity_ratio=1.0):
    block_factory = make_last_block_moe_factory(
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
    )
    return MoEConvNeXtWrapper(convnext_tiny(weights=None, num_classes=1000, block=block_factory))