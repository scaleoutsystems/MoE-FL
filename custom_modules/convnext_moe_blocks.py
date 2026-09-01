import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Optional
from torchvision.ops import Permute
from .routers import NoisyTopKRouter, ContextAwareNoisyTopKRouter, ContextRouterLogitCE
from .patch_dispatcher import PatchDispatcher

class MoEConvNeXtWrapper(nn.Module):
    def __init__(self, base: nn.Module, collect_expert_stats):
        super().__init__()
        self.base = base
        self._moe_modules = [m for m in self.base.modules() if hasattr(m, "aux_loss")]
        self.collect_expert_stats = collect_expert_stats
        
        
    @torch.compiler.disable
    def _set_context(self, context):
        for m in self._moe_modules:
            m._context = context

    def forward(self, x,context=None, return_aux: bool = False):
        # logits = self.base(x)
        # if not return_aux:
        #     return logits
        # return logits, self.get_aux_loss()
        self._set_context(context)
        logits = self.base(x)
        self._set_context(None)  # don't let stale context leak into the next call
        if not return_aux:
            return logits
        return logits, self.get_aux_loss()

    #To make torch.compile not cause recompiles
    @torch.compiler.disable
    def get_aux_loss(self):
        aux = None
        for m in self._moe_modules:
            a = m.aux_loss
            if a is None:
                continue
            aux = a if aux is None else aux + a
        if aux is None:
            aux = next(self.base.parameters()).new_zeros(())
        return aux
    
    def pop_expert_stats(self):
        stats = []
        #Append stats for each moe layer
        for m in self._moe_modules:
            stats.append(m.expert_stats.cpu())
            m.expert_stats.zero_()
        return stats
    

class MoECNBlock(nn.Module, PatchDispatcher): 
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        mlp_ratio: int,
        capacity_ratio : float,
        layer_scale: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        context_aware: bool = False,
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
        )
        
        if context_aware:
            self.router = ContextRouterLogitCE(dim,num_experts,top_k) 
        else:
            self.router = NoisyTopKRouter(dim, num_experts, top_k)
        self._context = None
        self.experts = nn.ModuleList([ConvNeXtExpert(dim, mlp_ratio) for _ in range(num_experts)])
        self.permute_back = Permute([0, 3, 1, 2])
        self.aux_loss = torch.tensor(0.0)
        self.capacity_ratio = capacity_ratio
        self.num_experts = num_experts
        self.register_buffer('expert_stats', torch.zeros(num_experts), persistent=False)

        #self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)

    #This module may not work well with torch.compile
    @torch.compiler.disable
    def forward(self, input):
        x = self.dwconv(input)  # (N, H, W, C)
        N, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # [T, C]
        
        context_tok = None
        if self._context is not None:
            # (N,) -> (T,): each image's label repeated over its H*W patches
            context_tok = self._context.to(x_flat.device).repeat_interleave(H * W)

        # topi, weights, priority, aux_loss = self.router(
        #     x_flat, context_label=context_tok, return_aux=self.training
        # )
        router_kwargs = {"return_aux": self.training}
        if context_tok is not None:
            router_kwargs["context_label"] = context_tok

        topi, weights, priority, aux_loss = self.router(x_flat, **router_kwargs)

        
        if self._context is not None:
            topi = torch.full_like(topi, int(self._context[0]))

        #topi, weights, priority, aux_loss = self.router(x_flat, return_aux=self.training)
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
        
        return input + moe_out + x_skip
    
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