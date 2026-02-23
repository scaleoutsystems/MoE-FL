import torch
import torch.nn as nn
import timm
from .routers import NoisyTopKRouter
from .patch_dispatcher import PatchDispatcher

class MoEViTWrapper(nn.Module):
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

class MoEViTBlock(nn.Module, PatchDispatcher): 
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        mlp_ratio: int,
        capacity_ratio: float
        ):
        super().__init__()
        self.num_experts = num_experts
        self.router = NoisyTopKRouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([ViTExpert(dim, mlp_ratio) for _ in range(num_experts)])
        self.aux_loss = torch.tensor(0.0)
        self.capacity_ratio = capacity_ratio

    def forward(self, x):
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        topi, weights, priority, aux_loss = self.router(x_flat, return_aux=self.training)
        order = torch.argsort(priority, descending=True)
        moe_flat = self.moe_route(x_flat, topi, weights, order)
        
        if aux_loss is None:
            self.aux_loss = x_flat.new_zeros(())
        else:
            self.aux_loss = aux_loss
        
        moe_out = moe_flat.view(B, N, C)
        
        return moe_out
    
class ViTExpert(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        return self.block(x)

def create_vit_s_moe(which="last2", num_experts=8, top_k=2, mlp_ratio=2,
                   capacity_ratio=1.0):
    """
    model: timm VisionTransformer
    which: 'every2' or 'last2' 
    """
    model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False
    
    ) 
    blocks = model.blocks
    L = len(blocks)

    if which == "every2":
        idxs = list(range(1, L, 2))   # 1,3,5,...
    elif which == "last2":
        idxs = [L - 2, L - 1]
    else:
        raise ValueError("which must be 'every2' or 'last2")

    for i in idxs:
        blk = blocks[i]
        dim = blk.mlp.fc1.in_features

        blk.mlp = MoEViTBlock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            num_experts=num_experts,
            top_k=top_k,
            capacity_ratio=capacity_ratio,
        )
    return model

def vit_s_moe_model_fn(which='last2',num_experts=8,top_k=2,mlp_ratio=2,capacity_ratio=1.0):
    vit_s_model = create_vit_s_moe(
        which = which,
        num_experts=num_experts,
        top_k=top_k,
        mlp_ratio=mlp_ratio,
        capacity_ratio=capacity_ratio,
    )
    return MoEViTWrapper(vit_s_model)