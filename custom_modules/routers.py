import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#Shazeer 2017 router
class NoisyTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        importance_coeff: float = 0.1,
        load_coeff: float = 0.1,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.importance_coeff = importance_coeff
        self.load_coeff = load_coeff
        self.eps = eps

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.noise = nn.Linear(dim, num_experts, bias=False)
        #Shazeer suggests initializing these as zeros
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.noise.weight)

    @staticmethod
    def _phi(z):
        # Standard normal CDF
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0))))
    
    @staticmethod
    def _cv_squared(x, eps = 1e-9):
        mean = x.mean()
        var = (x - mean).pow(2).mean()  
        return var / (mean * mean + eps)

    def _importance_loss(self, gates_full):
        importance = gates_full.sum(dim=0)      
        return self.importance_coeff * self._cv_squared(importance, self.eps)
    
    def _load_loss(self, logits_clean, noise_std, noisy_logits):
        B, E = logits_clean.shape
        k = self.top_k

        # kth and (k+1)th largest per token from noisy logits (matches routing)
        top_vals, top_idx = noisy_logits.topk(k + 1, dim=-1)
        v_k = top_vals[:, k - 1]
        v_kplus = top_vals[:, k]

        in_topk = torch.zeros(B, E, device=noisy_logits.device, dtype=torch.bool)
        in_topk.scatter_(1, top_idx[:, :k], True)

        kth_excl = torch.where(
            in_topk,
            v_kplus.unsqueeze(-1).expand_as(noisy_logits),
            v_k.unsqueeze(-1).expand_as(noisy_logits),
        )

        z = (logits_clean - kth_excl) / (noise_std + self.eps)
        P = self._phi(z)
        load = P.sum(dim=0)
        return self.load_coeff * self._cv_squared(load, self.eps)
    
    def forward(self, x, return_aux=True):
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        logits = self.gate(x2)  # clean logits
        noise_std = F.softplus(self.noise(x2)) + self.eps

        # Only add noise during training
        if self.training:
            noisy_logits = logits + torch.randn_like(logits) * noise_std
        else:
            noisy_logits = logits

        # Shazeer KeepTopK(noisy_logits) then softmax
        topv, topi = noisy_logits.topk(self.top_k, dim=-1)
        masked = noisy_logits.new_full(noisy_logits.shape, float("-inf"))
        masked.scatter_(1, topi, topv)

        gates_full = torch.softmax(masked, dim=-1)
        weights = gates_full.gather(1, topi)

        # BPR priority
        priority = weights.max(dim=-1).values

        # aux losses only in training
        aux_loss = None
        if return_aux and self.training and (self.importance_coeff != 0.0 or self.load_coeff != 0.0):
            aux = logits.new_zeros(())
            if self.importance_coeff != 0.0:
                aux = aux + self._importance_loss(gates_full)   # uses sparse gates
            if self.load_coeff != 0.0:
                aux = aux + self._load_loss(logits, noise_std, noisy_logits)  # uses noisy threshold
            aux_loss = aux

        leading = orig_shape[:-1]
        topi = topi.view(*leading, self.top_k)
        weights = weights.view(*leading, self.top_k)
        priority = priority.view(*leading)

        return topi, weights, priority, aux_loss
    
    
#Switch router from paper
class SwitchRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        load_coeff: float = 0.1,
        eps: float = 1e-9,
        jitter_eps: float = 1e-2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 1 #Routing designed for Top-1
        self.load_coeff = load_coeff
        self.eps = eps
        self.jitter_eps = jitter_eps

        self.gate = nn.Linear(dim, num_experts, bias=False)
        #Initialize with small weights 
        std = (0.1 / dim) ** 0.5
        nn.init.normal_(self.gate.weight, mean=0.0, std=std)
        
    def _load_loss(self, probs, top1):
        B, E = probs.shape

        # f_i: fraction of tokens routed to expert i (hard counts)
        counts = torch.bincount(top1, minlength=E).float()        
        f = counts / (B + self.eps)                                   
        # p_i: mean probability mass for expert i 
        p = probs.mean(dim=0)                                

        return self.load_coeff * (E * torch.sum(f * p))

    
    def forward(self, x, return_aux=True):
        T, _ = x.shape
        
        if self.training and self.jitter_eps > 0:
            # multiplicative uniform noise for exploration
            x = x * (1.0 + (2.0 * torch.rand_like(x) - 1.0) * self.jitter_eps)
        
        #Compute probabiliities
        logits = self.gate(x)                 
        probs = F.softmax(logits, dim=-1)     

        #Get Top-1
        top1 = torch.argmax(probs, dim=-1)    
        top1_prob = probs.gather(1, top1[:, None]).squeeze(1) 
        topi = top1[:, None]                 
        weights = top1_prob[:, None]         

        #Randomize so overflow does not always drop the same tokens
        priority = -torch.arange(T, device=x.device, dtype=x.dtype)

        #Add aux_loss if set to true
        aux_loss = self._load_loss(probs, top1) if (return_aux and self.training) else None
        
        return topi, weights, priority, aux_loss