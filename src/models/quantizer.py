import torch
import torch.nn as nn
import torch.nn.functional as F


class VQQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = commitment_cost
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())

    def forward(self, z):
        # z: (B, D, H, W) -> flatten to (N, D)
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Distances: ||z - e||^2
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        indices = dist.argmin(dim=1)
        onehot = F.one_hot(indices, self.K).float()
        z_q = onehot @ self.embedding.weight

        # EMA update
        if self.training:
            with torch.no_grad():
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                n = self.ema_count.sum()
                smoothed = (self.ema_count + self.eps) / (n + self.K * self.eps) * n
                dw = onehot.t() @ z_flat
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                self.embedding.weight.data.copy_(self.ema_weight / smoothed.unsqueeze(1))

        # Losses
        commitment = self.beta * F.mse_loss(z_flat, z_q.detach())
        # Straight-through
        z_q_st = z_flat + (z_q - z_flat).detach()
        z_q_st = z_q_st.view(B, H, W, D).permute(0, 3, 1, 2)

        # Perplexity
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices = indices.view(B, H, W)
        return z_q_st, commitment, indices, perplexity
