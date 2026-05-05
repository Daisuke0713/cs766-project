"""
Generate report artifacts: loss curves, reconstruction images, metrics table.

Usage:
    python evaluate.py --vqvae-dir out/vqvae --wvq-dir out/wvq --output report/
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import VQVAE, WaveletVQVAE


def get_val_loader(dataset, data_dir, batch_size=64, image_size=32):
    if dataset == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        val = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif dataset == "cifar100":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        val = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    elif dataset == "stl10":
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        val = datasets.STL10(data_dir, split="test", download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2)


def load_model(model_dir, device):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    if cfg["model"] == "vqvae":
        model = VQVAE(
            hidden=cfg["hidden"], res_hidden=cfg["res_hidden"],
            embed_dim=cfg["embed_dim"], num_embeddings=cfg["num_embeddings"],
            commitment_cost=cfg["commitment_cost"], decay=cfg["decay"], num_res=cfg["num_res"],
        )
    else:
        cb = {
            "LL2": cfg["cb_ll2"], "LH2": cfg["cb_lh2"], "HL2": cfg["cb_hl2"], "HH2": cfg["cb_hh2"],
            "LH1": cfg["cb_lh1"], "HL1": cfg["cb_hl1"], "HH1": cfg["cb_hh1"],
        }
        model = WaveletVQVAE(
            hidden=cfg["hidden"], res_hidden=cfg["res_hidden"],
            embed_dim=cfg["embed_dim"], codebook_sizes=cb,
            commitment_cost=cfg["commitment_cost"], decay=cfg["decay"], num_res=cfg["num_res"],
        )
    ckpt = os.path.join(model_dir, "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(model_dir, "final.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    return model, cfg


def compute_psnr(x, x_rec):
    mse = F.mse_loss(x_rec, x).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim_simple(x, x_rec):
    # simple windowed SSIM
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, 3, stride=1, padding=1)
    mu_y = F.avg_pool2d(x_rec, 3, stride=1, padding=1)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(x * x, 3, stride=1, padding=1) - mu_x2
    sigma_y2 = F.avg_pool2d(x_rec * x_rec, 3, stride=1, padding=1) - mu_y2
    sigma_xy = F.avg_pool2d(x * x_rec, 3, stride=1, padding=1) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean().item()


@torch.no_grad()
def eval_metrics(model, loader, device, num_batches=50):
    psnr_vals, ssim_vals, recon_vals = [], [], []
    for i, (x, _) in enumerate(loader):
        if i >= num_batches:
            break
        x = x.to(device)
        x_rec, _, _, _ = model(x)
        recon_vals.append(F.mse_loss(x_rec, x).item())
        psnr_vals.append(compute_psnr(x, x_rec))
        ssim_vals.append(compute_ssim_simple(x, x_rec))
    return {
        "recon_mse": np.mean(recon_vals),
        "psnr": np.mean(psnr_vals),
        "ssim": np.mean(ssim_vals),
    }


@torch.no_grad()
def eval_codebook_utilization(model, loader, device, num_batches=50):
    if isinstance(model, VQVAE):
        all_idx = []
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            idx = model.encode(x.to(device))
            all_idx.append(idx.cpu().reshape(-1))
        all_idx = torch.cat(all_idx)
        used = len(torch.unique(all_idx))
        total = model.vq.K
        return {"codebook": f"{used}/{total} ({100*used/total:.1f}%)"}
    else:
        result = {}
        all_idx = {n: [] for n in model.cb_sizes}
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            indices = model.encode(x.to(device))
            for n in indices:
                all_idx[n].append(indices[n].cpu().reshape(-1))
        for n in all_idx:
            cat = torch.cat(all_idx[n])
            used = len(torch.unique(cat))
            total = model.cb_sizes[n]
            result[n] = f"{used}/{total} ({100*used/total:.1f}%)"
        return result


def plot_loss_curves(dirs, labels, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for d, label in zip(dirs, labels):
        log = json.load(open(os.path.join(d, "log.json")))
        eps = [e["epoch"] for e in log]
        train_recon = [e["train"]["recon_loss"] for e in log]
        val_recon = [e["val"]["recon_loss"] for e in log]
        axes[0].plot(eps, train_recon, label=f"{label} (train)")
        axes[0].plot(eps, val_recon, "--", label=f"{label} (val)")
        train_vq = [e["train"]["vq_loss"] for e in log]
        val_vq = [e["val"]["vq_loss"] for e in log]
        axes[1].plot(eps, train_vq, label=f"{label} (train)")
        axes[1].plot(eps, val_vq, "--", label=f"{label} (val)")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Recon Loss (MSE)")
    axes[0].set_title("Reconstruction Loss")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("VQ Loss")
    axes[1].set_title("VQ Commitment Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves: {output_path}")


def plot_reconstructions(models_and_labels, loader, device, output_path, num_images=8):
    x, _ = next(iter(loader))
    x = x[:num_images].to(device)

    n_models = len(models_and_labels)
    fig, axes = plt.subplots(1 + n_models, num_images, figsize=(num_images * 2, (1 + n_models) * 2))

    def to_img(t):
        return np.clip(t.cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1)

    for i in range(num_images):
        axes[0, i].imshow(to_img(x[i]))
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11)

    for row, (model, label) in enumerate(models_and_labels, 1):
        with torch.no_grad():
            x_rec, _, _, _ = model(x)
        for i in range(num_images):
            axes[row, i].imshow(to_img(x_rec[i]))
            axes[row, i].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstructions: {output_path}")


def plot_wavelet_subbands(model, loader, device, output_path):
    if not isinstance(model, WaveletVQVAE):
        return
    x, _ = next(iter(loader))
    x = x[:1].to(device)

    with torch.no_grad():
        ll1, lh1, hl1, hh1 = model.dwt(x)
        ll2, lh2, hl2, hh2 = model.dwt(ll1)

    def norm(t):
        t = t[0].mean(0).cpu().numpy()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return t

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(np.clip(x[0].cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1))
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(norm(ll1), cmap="gray"); axes[0, 1].set_title("LL1")
    axes[0, 2].imshow(norm(lh1), cmap="gray"); axes[0, 2].set_title("LH1")
    axes[0, 3].imshow(norm(hl1), cmap="gray"); axes[0, 3].set_title("HL1")
    axes[1, 0].imshow(norm(hh1), cmap="gray"); axes[1, 0].set_title("HH1")
    axes[1, 1].imshow(norm(ll2), cmap="gray"); axes[1, 1].set_title("LL2")
    axes[1, 2].imshow(norm(lh2), cmap="gray"); axes[1, 2].set_title("LH2")
    axes[1, 3].imshow(norm(hl2), cmap="gray"); axes[1, 3].set_title("HL2")
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle("2-Level Haar Wavelet Decomposition", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved subband visualization: {output_path}")


def print_metrics_table(results):
    header = "| Model | MSE | PSNR (dB) | SSIM | Params | CB Params | CB Utilization |"
    sep = "|---|---|---|---|---|---|---|"
    print("\n" + header)
    print(sep)
    for r in results:
        util_str = r["utilization"] if isinstance(r["utilization"], str) else json.dumps(r["utilization"])
        print(f"| {r['name']} | {r['mse']:.6f} | {r['psnr']:.2f} | {r['ssim']:.4f} "
              f"| {r['params']:,} | {r['cb_params']:,} | {util_str} |")
    print()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    os.makedirs(args.output, exist_ok=True)

    dirs, labels = [], []
    if args.vqvae_dir:
        dirs.append(args.vqvae_dir); labels.append("VQ-VAE")
    if args.wvq_dir:
        dirs.append(args.wvq_dir); labels.append("WVQ")

    loader = get_val_loader(args.dataset, args.data_dir, image_size=args.image_size)

    # Load models
    models_labels = []
    results = []
    for d, label in zip(dirs, labels):
        model, cfg = load_model(d, device)
        models_labels.append((model, label))
        metrics = eval_metrics(model, loader, device)
        util = eval_codebook_utilization(model, loader, device)
        nparams = sum(p.numel() for p in model.parameters())
        results.append({
            "name": label,
            "mse": metrics["recon_mse"],
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "params": nparams,
            "cb_params": model.codebook_params(),
            "utilization": util,
        })

    # Loss curves
    if len(dirs) > 0:
        plot_loss_curves(dirs, labels, os.path.join(args.output, "loss_curves.png"))

    # Reconstructions
    if len(models_labels) > 0:
        plot_reconstructions(models_labels, loader, device,
                             os.path.join(args.output, "reconstructions.png"))

    # Wavelet subband visualization
    for model, label in models_labels:
        if isinstance(model, WaveletVQVAE):
            plot_wavelet_subbands(model, loader, device,
                                  os.path.join(args.output, "wavelet_subbands.png"))

    # Metrics table
    print_metrics_table(results)

    # Save metrics as json
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved metrics: {os.path.join(args.output, 'metrics.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vqvae-dir", type=str, default=None)
    p.add_argument("--wvq-dir", type=str, default=None)
    p.add_argument("--output", type=str, default="report")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10"], default="cifar10")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    main(p.parse_args())
