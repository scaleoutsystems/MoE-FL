import torch

#Mixin class for shared patch routing functionality
class PatchDispatcher:     
    def moe_route(self, x_flat, topi, weights, order):
        T, C = x_flat.shape
        K = topi.shape[1]
        E = self.num_experts
        device = x_flat.device

        # Build flattened routes in BPR order
        token_idx = order.repeat(K) 

        # expert ids in round order (rank-0 for all tokens, then rank-1, ...)
        expert_id = topi[order].transpose(0, 1).reshape(-1)  

        # gate weights aligned with expert_id
        gate_w = weights[order].transpose(0, 1).reshape(-1, 1) 

        # Group by expert id while preserving BPR order inside each expert
        sort_perm = torch.argsort(expert_id, stable=True)
        token_idx = token_idx[sort_perm]
        expert_id = expert_id[sort_perm]
        gate_w = gate_w[sort_perm]

        #Be = int(round((K * T * self.capacity_ratio) / E))
        Be = int((K * T * self.capacity_ratio) / E + 0.5)
        assert(Be > 0)

        # Find start index of each expert chunk
        counts = torch.bincount(expert_id, minlength=E)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        # Position of each route within its expert chunk
        M = expert_id.numel()
        pos_in_chunk = torch.arange(M, device=device) - starts[expert_id]

        # Keep only first Be routes per expert (highest priority due to BPR ordering)
        keep = pos_in_chunk < Be

        token_idx = token_idx[keep]
        expert_id = expert_id[keep]
        gate_w = gate_w[keep]

        out = x_flat.new_zeros(T, C)

        # Recompute chunk boundaries after dropping overflow routes
        counts = torch.bincount(expert_id, minlength=E)
        offsets = torch.cumsum(counts, dim=0)
        starts  = offsets - counts
        
        if self.collect_expert_stats:
            self._last_routing = (expert_id.cpu(), token_idx.cpu(), T)
        
        #Build padded input tensor
        # slot[e, pos] = which position within expert e's chunk this route occupies
        pos_in_chunk = torch.arange(token_idx.numel(), device=device) - starts[expert_id]

        #Gather tokens into padded buffer zero padding fills unused slots
        x_in = x_flat.new_zeros(E, Be, C)                       
        x_in[expert_id, pos_in_chunk] = x_flat[token_idx]      

        # Also store gate weights in the same layout for later
        g_in = gate_w.new_zeros(E, Be, 1)                      
        g_in[expert_id, pos_in_chunk] = gate_w

        #Batched expert forward 
        w1 = torch.stack([exp.block[0].weight.t() for exp in self.experts])  
        b1 = torch.stack([exp.block[0].bias       for exp in self.experts])  
        w2 = torch.stack([exp.block[2].weight.t() for exp in self.experts])  
        b2 = torch.stack([exp.block[2].bias       for exp in self.experts]) 
        h = torch.bmm(x_in, w1) + b1.unsqueeze(1)   
        h = torch.nn.functional.gelu(h)
        y = torch.bmm(h, w2) + b2.unsqueeze(1)       
        # Apply gate weights
        y = y * g_in                                  

        #Scatter back
        out = x_flat.new_zeros(T, C)
        out.index_add_(0, token_idx, y[expert_id, pos_in_chunk])

        return out   
class MoEStatsCollector:
    def __init__(self, model, num_classes: int):
        self.num_classes = num_classes
        self.layers: dict[str, PatchDispatcher] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PatchDispatcher)
        }

    def run(self, model, val_loader, device):
        E = next(iter(self.layers.values())).num_experts
        accum = {
            name: {
                "tokens_kept":       torch.zeros(E, dtype=torch.long),
                "total_kept":        0,
                "class_expert_hits": torch.zeros(self.num_classes, E, dtype=torch.long),
            }
            for name in self.layers
        }
        
        for layer in self.layers.values():
            layer.collect_expert_stats = True

        model.eval()
        with torch.no_grad():
            for x, labels in val_loader:
                model(x.to(device))
                labels_cpu = labels.cpu()

                for name, layer in self.layers.items():
                    expert_id, token_idx, T = layer._last_routing
                    patches_per_image = T // labels_cpu.shape[0]
                    image_idx = token_idx // patches_per_image                   
                    a = accum[name]
                    a["tokens_kept"].scatter_add_(0, expert_id, torch.ones_like(expert_id))
                    a["total_kept"] += expert_id.numel()
                    token_labels = labels_cpu[image_idx]
                    idx = token_labels * layer.num_experts + expert_id
                    a["class_expert_hits"].view(-1).scatter_add_(0, idx, torch.ones_like(idx))

        return {
            name: self._compute_stats(accum[name], layer.num_experts)
            for name, layer in self.layers.items()
        }

    def _compute_stats(self, a, E):
        expert_activation_pct = (a["tokens_kept"] / (a["total_kept"] + 1e-9)) * 100

        class_totals = a["class_expert_hits"].sum(dim=1, keepdim=True)
        class_expert_pct = (a["class_expert_hits"] / (class_totals + 1e-9)) * 100

        return {
            "expert_activation_pct":       expert_activation_pct,       # (E,)
            "class_expert_activation_pct": class_expert_pct,            # (num_classes, E)
        }