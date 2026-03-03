import torch

#Mixin class for shared patch routing functionality
class PatchDispatcher:     
    #TODO: If found to be bottleneck, rewrite for speed
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
        
        if self.collect_expert_stats:
            self._last_routing = (expert_id[keep].cpu(), token_idx[keep].cpu(), T)

        token_idx = token_idx[keep]
        expert_id = expert_id[keep]
        gate_w = gate_w[keep]

        out = x_flat.new_zeros(T, C)

        # Recompute chunk boundaries after dropping overflow routes
        counts = torch.bincount(expert_id, minlength=E)
        offsets = torch.cumsum(counts, dim=0)

        start = 0
        for e, expert in enumerate(self.experts):
            end = int(offsets[e])
            if end == start:
                continue

            idx_e = token_idx[start:end]        # tokens kept for this expert
            x_e = x_flat.index_select(0, idx_e) # gather tokens
            y_e = expert(x_e) * gate_w[start:end]
            out.index_add_(0, idx_e, y_e)
            start = end

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