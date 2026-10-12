import torch

#Mixin class for shared patch routing functionality
class PatchDispatcher:     
    def moe_route(self, x_flat, topi, weights, order):
        T, C = x_flat.shape
        K = topi.shape[1]
        E = self.num_experts
        device = x_flat.device

        token_idx = order.repeat(K)
        expert_id = topi[order].transpose(0, 1).reshape(-1)
        gate_w = weights[order].transpose(0, 1).reshape(-1, 1)

        sort_perm = torch.argsort(expert_id, stable=True)
        token_idx = token_idx[sort_perm]
        expert_id = expert_id[sort_perm]
        gate_w = gate_w[sort_perm]

        out = x_flat.new_zeros(T, C)

        if self.bpr:
            counts = torch.bincount(expert_id, minlength=E)
            offsets = torch.cumsum(counts, dim=0)
            starts = offsets - counts

            Be = int((K * T * self.capacity_ratio) / E + 0.5)
            assert Be > 0

            M = expert_id.numel()
            pos_in_chunk = torch.arange(M, device=device) - starts[expert_id]
            keep = pos_in_chunk < Be

            token_idx = token_idx[keep]
            expert_id = expert_id[keep]
            gate_w = gate_w[keep]

            # Recompute AFTER dropping — this is what actually gets processed
            counts = torch.bincount(expert_id, minlength=E)
            offsets = torch.cumsum(counts, dim=0)
            starts = offsets - counts
            pos_in_chunk = torch.arange(token_idx.numel(), device=device) - starts[expert_id]

            if self.training:
                self.expert_stats += counts.detach()
            else:
                self._last_routing = (expert_id.cpu(), token_idx.cpu(), T)

            x_in = x_flat.new_zeros(E, Be, C)
            x_in[expert_id, pos_in_chunk] = x_flat[token_idx]
            g_in = gate_w.new_zeros(E, Be, 1)
            g_in[expert_id, pos_in_chunk] = gate_w

            w1 = torch.stack([exp.block[0].weight.t() for exp in self.experts])
            b1 = torch.stack([exp.block[0].bias       for exp in self.experts])
            w2 = torch.stack([exp.block[2].weight.t() for exp in self.experts])
            b2 = torch.stack([exp.block[2].bias       for exp in self.experts])
            h = torch.bmm(x_in, w1) + b1.unsqueeze(1)
            h = torch.nn.functional.gelu(h)
            y = torch.bmm(h, w2) + b2.unsqueeze(1)
            y = y * g_in

            out.index_add_(0, token_idx, y[expert_id, pos_in_chunk])

        else:
            counts = torch.bincount(expert_id, minlength=E)
            if self.training:
                self.expert_stats += counts.detach()
            else:
                self._last_routing = (expert_id.cpu(), token_idx.cpu(), T)

            for e in range(E):
                mask = expert_id == e
                if not mask.any():
                    continue
                idx = token_idx[mask]
                g = gate_w[mask]
                y = self.experts[e](x_flat[idx]) * g
                out.index_add_(0, idx, y)

        return out