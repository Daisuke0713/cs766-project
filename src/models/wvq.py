import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual import ResidualStack
from .quantizer import VQQuantizer


class HaarDWT2D(nn.Module):

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[1, 1], [-1, -1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[1, -1], [1, -1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        B, C, H, W = x.shape
        y = F.conv2d(x.reshape(B * C, 1, H, W), self.filters, stride=2)
        y = y.view(B, C, 4, H // 2, W // 2)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]

    def inverse(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        y = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, H, W)
        x = F.conv_transpose2d(y, self.filters, stride=2)
        return x.reshape(B, C, H * 2, W * 2)


class SubbandEncoder(nn.Module):
    def __init__(self, in_ch, hidden, res_hidden, embed_dim, num_res=2, downsample=False):
        super().__init__()
        layers = []
        if downsample:
            layers += [nn.Conv2d(in_ch, hidden // 2, 4, stride=2, padding=1), nn.ReLU(True)]
            layers += [nn.Conv2d(hidden // 2, hidden, 3, padding=1)]
        else:
            layers += [nn.Conv2d(in_ch, hidden // 2, 3, padding=1), nn.ReLU(True)]
            layers += [nn.Conv2d(hidden // 2, hidden, 3, padding=1)]
        self.conv = nn.Sequential(*layers)
        self.res = ResidualStack(hidden, res_hidden, num_res)
        self.proj = nn.Conv2d(hidden, embed_dim, 1)

    def forward(self, x):
        return self.proj(self.res(self.conv(x)))


class SubbandDecoder(nn.Module):
    def __init__(self, out_ch, hidden, res_hidden, embed_dim, num_res=2, upsample=False):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, hidden, 1)
        self.res = ResidualStack(hidden, res_hidden, num_res)
        if upsample:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden, hidden // 2, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(hidden // 2, out_ch, 4, stride=2, padding=1),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden, hidden // 2, 3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(hidden // 2, out_ch, 3, padding=1),
            )

    def forward(self, z_q):
        return self.conv(self.res(self.proj(z_q)))


# Subband names and default codebook sizes
SUBBANDS = ["LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"]
DEFAULT_CB = {"LL2": 512, "LH2": 128, "HL2": 128, "HH2": 32, "LH1": 256, "HL1": 256, "HH1": 64}


class WaveletVQVAE(nn.Module):
    """2-level Haar Wavelet VQ-VAE with per-subband codebooks."""

    def __init__(
        self,
        in_channels=3,
        hidden=64,
        res_hidden=32,
        embed_dim=32,
        codebook_sizes=None,
        commitment_cost=0.25,
        decay=0.99,
        num_res=2,
    ):
        super().__init__()
        self.dwt = HaarDWT2D()
        cb = codebook_sizes or DEFAULT_CB

        # Level-1 subbands are 16x16, downsample to 8x8 latent
        # Level-2 subbands are 8x8, keep as 8x8 latent
        is_level1 = lambda n: n.endswith("1")

        self.encoders = nn.ModuleDict()
        self.vqs = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.cb_sizes = {}

        for name in SUBBANDS:
            ds = is_level1(name)
            K = cb[name]
            self.cb_sizes[name] = K
            self.encoders[name] = SubbandEncoder(
                in_channels, hidden, res_hidden, embed_dim, num_res, downsample=ds
            )
            self.vqs[name] = VQQuantizer(K, embed_dim, commitment_cost, decay)
            self.decoders[name] = SubbandDecoder(
                in_channels, hidden, res_hidden, embed_dim, num_res, upsample=ds
            )

    def forward(self, x):
        # Level 1
        ll1, lh1, hl1, hh1 = self.dwt(x)
        # Level 2 on LL1
        ll2, lh2, hl2, hh2 = self.dwt(ll1)

        subs = {"LL2": ll2, "LH2": lh2, "HL2": hl2, "HH2": hh2, "LH1": lh1, "HL1": hl1, "HH1": hh1}

        rec = {}
        total_vq_loss = 0.0
        ppl_dict = {}
        idx_dict = {}

        for name in SUBBANDS:
            z = self.encoders[name](subs[name])
            z_q, loss, indices, ppl = self.vqs[name](z)
            rec[name] = self.decoders[name](z_q)
            total_vq_loss += loss
            ppl_dict[name] = ppl
            idx_dict[name] = indices

        total_vq_loss /= len(SUBBANDS)

        # Inverse: reconstruct LL1 from level-2 subbands, then full image
        ll1_rec = self.dwt.inverse(rec["LL2"], rec["LH2"], rec["HL2"], rec["HH2"])
        x_recon = self.dwt.inverse(ll1_rec, rec["LH1"], rec["HL1"], rec["HH1"])

        return x_recon, total_vq_loss, idx_dict, ppl_dict

    def encode(self, x):
        ll1, lh1, hl1, hh1 = self.dwt(x)
        ll2, lh2, hl2, hh2 = self.dwt(ll1)
        subs = {"LL2": ll2, "LH2": lh2, "HL2": hl2, "HH2": hh2, "LH1": lh1, "HL1": hl1, "HH1": hh1}
        indices = {}
        for name in SUBBANDS:
            z = self.encoders[name](subs[name])
            _, _, idx, _ = self.vqs[name](z)
            indices[name] = idx
        return indices

    def codebook_params(self):
        total = 0
        for name in SUBBANDS:
            total += self.cb_sizes[name] * self.vqs[name].D
        return total