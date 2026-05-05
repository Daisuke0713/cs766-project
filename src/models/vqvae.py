import torch.nn as nn
from .residual import ResidualStack
from .quantizer import VQQuantizer


class Encoder(nn.Module):
    def __init__(self, in_ch, hidden, res_hidden, embed_dim, num_res=2):
        super().__init__()
        # 32x32 -> 16x16 -> 8x8
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden // 2, hidden, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.res = ResidualStack(hidden, res_hidden, num_res)
        self.proj = nn.Conv2d(hidden, embed_dim, 1)

    def forward(self, x):
        return self.proj(self.res(self.net(x)))


class Decoder(nn.Module):
    def __init__(self, out_ch, hidden, res_hidden, embed_dim, num_res=2):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, hidden, 1)
        self.res = ResidualStack(hidden, res_hidden, num_res)
        # 8x8 -> 16x16 -> 32x32
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden // 2, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, z_q):
        return self.net(self.res(self.proj(z_q)))


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden=128,
        res_hidden=32,
        embed_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        num_res=2,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden, res_hidden, embed_dim, num_res)
        self.vq = VQQuantizer(num_embeddings, embed_dim, commitment_cost, decay)
        self.decoder = Decoder(in_channels, hidden, res_hidden, embed_dim, num_res)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices, perplexity

    def encode(self, x):
        z = self.encoder(x)
        _, _, indices, _ = self.vq(z)
        return indices

    def codebook_params(self):
        return self.vq.K * self.vq.D
