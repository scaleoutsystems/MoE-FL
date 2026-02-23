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