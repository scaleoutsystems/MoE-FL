import torch
import torch.nn as nn
from functools import partial
from typing import Callable, Optional
from torchvision.ops import Permute
from .routers import NoisyTopKRouter, ContextRouterLogitCE
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

    #Restricts dispatch to a subset of experts (e.g. the experts a federated
    #client owns). None removes the restriction.
    @torch.compiler.disable
    def set_allowed_experts(self, allowed_experts):
        for m in self._moe_modules:
            m.allowed_experts = allowed_experts

    #inputs are passed through to base: x for classification, (images, targets) for detection
    def forward(self, *inputs, context=None, return_aux: bool = False):
        if context is not None:
            self._set_context(context)
        out = self.base(*inputs)
        self._set_context(None)  # don't let stale context leak into the next call
        if not return_aux:
            return out
        return out, self.get_aux_loss()

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
            #clone: on CPU .cpu() returns the buffer itself, which zero_() would wipe
            stats.append(m.expert_stats.detach().cpu().clone())
            m.expert_stats.zero_()
        return stats


def _remap_disallowed_experts(topi, logits, allowed_expert_ids):
    T, k = topi.shape
    E = logits.shape[1]
    device = topi.device

    allowed_mask = torch.zeros(E, dtype=torch.bool, device=device)
    allowed_ids = torch.as_tensor(list(allowed_expert_ids), device=device, dtype=torch.long)
    allowed_mask[allowed_ids] = True

    assert allowed_ids.numel() >= k, (
        f"allowed_expert_ids has {allowed_ids.numel()} experts, fewer than top_k={k}"
    )

    masked_logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
    new_topi = topi.clone()

    for slot in range(k):
        needs_fix = ~allowed_mask[topi[:, slot]]
        if not needs_fix.any():
            continue
        cand_logits = masked_logits.clone()
        for other in range(k):
            if other == slot:
                continue
            cand_logits.scatter_(1, new_topi[:, other:other + 1], float("-inf"))
        best_allowed = cand_logits.argmax(dim=-1)
        new_topi[needs_fix, slot] = best_allowed[needs_fix]

    return new_topi


def remap_dispatch_only(topi, weights, priority, logits, allowed_experts):
    """
    Default expert-restriction strategy: reroutes disallowed picks to the
    best allowed alternative for dispatch only. 
    """
    new_topi = _remap_disallowed_experts(topi, logits, allowed_experts)
    return new_topi, weights, priority


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
        bpr: bool = True,
        allowed_experts: Optional[list] = None,
        restriction_fn: Callable = remap_dispatch_only,
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
        self.context_aware = context_aware
        self._context = None
        self.experts = nn.ModuleList([ConvNeXtExpert(dim, mlp_ratio) for _ in range(num_experts)])
        self.permute_back = Permute([0, 3, 1, 2])
        self.aux_loss = torch.tensor(0.0)
        self.capacity_ratio = capacity_ratio
        self.num_experts = num_experts
        self.bpr = bpr
        self.allowed_experts = allowed_experts
        self.restriction_fn = restriction_fn
        self.register_buffer('expert_stats', torch.zeros(num_experts), persistent=False)

        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)

    #This module may not work well with torch.compile
    @torch.compiler.disable
    def forward(self, input):
        x = self.dwconv(input)  # (N, H, W, C)
        N, H, W, C = x.shape
        x_flat = x.reshape(-1, C)  # [T, C]
        
        context_tok = None
        #Only the context router uses labels, others ignore any context they are given
        if self._context is not None and self.context_aware:
            # (N,) -> (T,): each image's context label repeated over its H*W patches
            context_tok = self._context.to(x_flat.device).repeat_interleave(H * W)

        router_kwargs = {"return_aux": self.training}
        if context_tok is not None:
            router_kwargs["context_label"] = context_tok

        topi, weights, priority, aux_loss, logits = self.router(x_flat, **router_kwargs)

        if self.allowed_experts is not None:
            topi, weights, priority = self.restriction_fn(
                topi, weights, priority, logits, self.allowed_experts
            )

        if self.bpr:
            order = torch.argsort(priority, descending=True)
        else:
            order = torch.arange(x_flat.shape[0], device=x_flat.device)
        moe_flat = self.moe_route(x_flat, topi, weights, order)  # [T, C]
        
        if aux_loss is None:
           self.aux_loss = x_flat.new_zeros(())
        else:
           self.aux_loss = aux_loss
            
        #Additional skip connection described in paper       
        x_skip = self.permute_back(x)

        moe_out = moe_flat.view(N, H, W, C)
        moe_out = self.permute_back(moe_out)  # (N, C, H, W)
        
        #Layer scale on the whole branch (experts + skip), like CNBlock
        return input + self.layer_scale * (moe_out + x_skip)
    
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