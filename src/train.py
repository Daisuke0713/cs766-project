"""
Train vanilla VQ-VAE or Wavelet VQ-VAE on CIFAR-10, CIFAR-100, or STL-10.

Usage:
    python train.py --model vqvae --output out/vqvae --dataset cifar10
    python train.py --model wvq   --output out/wvq   --dataset stl10 --image-size 64
"""
import os
import json
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import VQVAE, WaveletVQVAE


def get_loaders(dataset, data_dir, batch_size, image_size=32, workers=4):
    if dataset == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        val = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
    elif dataset == "cifar100":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=tf)
        val = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    elif dataset == "stl10":
        tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
        ])
        train = datasets.STL10(data_dir, split="train", download=True, transform=tf)
        val = datasets.STL10(data_dir, split="test", download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    tl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    vl = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return tl, vl


def build_model(args):
    if args.model == "vqvae":
        return VQVAE(
            in_channels=3,
            hidden=args.hidden,
            res_hidden=args.res_hidden,
            embed_dim=args.embed_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            decay=args.decay,
            num_res=args.num_res,
        )
    else:
        cb = {
            "LL2": args.cb_ll2, "LH2": args.cb_lh2, "HL2": args.cb_hl2, "HH2": args.cb_hh2,
            "LH1": args.cb_lh1, "HL1": args.cb_hl1, "HH1": args.cb_hh1,
        }
        return WaveletVQVAE(
            in_channels=3,
            hidden=args.hidden,
            res_hidden=args.res_hidden,
            embed_dim=args.embed_dim,
            codebook_sizes=cb,
            commitment_cost=args.commitment_cost,
            decay=args.decay,
            num_res=args.num_res,
        )


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_recon, total_vq, n = 0.0, 0.0, 0

    ctx = torch.no_grad if not train else torch.enable_grad
    with ctx():
        for x, _ in loader:
            x = x.to(device)
            x_rec, vq_loss, _, _ = model(x)
            recon_loss = F.mse_loss(x_rec, x)

            if train:
                loss = recon_loss + vq_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            n += 1

    return {"recon_loss": total_recon / n, "vq_loss": total_vq / n}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Device: {device}")

    tl, vl = get_loaders(args.dataset, args.data_dir, args.batch_size, image_size=args.image_size)
    model = build_model(args).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    cb_params = model.codebook_params()
    print(f"Model: {args.model} | Params: {nparams:,} | Codebook params: {cb_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    log = []
    best_val = float("inf")

    for ep in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_m = run_epoch(model, tl, optimizer, device, train=True)
        val_m = run_epoch(model, vl, None, device, train=False)
        log.append({"epoch": ep, "train": train_m, "val": val_m})

        if ep % args.log_every == 0 or ep == args.epochs:
            tqdm.write(
                f"Ep {ep:3d} | train recon={train_m['recon_loss']:.5f} vq={train_m['vq_loss']:.5f} "
                f"| val recon={val_m['recon_loss']:.5f} vq={val_m['vq_loss']:.5f}"
            )

        val_total = val_m["recon_loss"] + val_m["vq_loss"]
        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), os.path.join(args.output, "best.pt"))

    torch.save(model.state_dict(), os.path.join(args.output, "final.pt"))
    with open(os.path.join(args.output, "log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"Done. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vqvae", "wvq"], required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "stl10"], default="cifar10")
    p.add_argument("--image-size", type=int, default=32, help="Resize for stl10; ignored for cifar")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=10)

    # Architecture
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--res-hidden", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--num-res", type=int, default=2)
    p.add_argument("--commitment-cost", type=float, default=0.25)
    p.add_argument("--decay", type=float, default=0.99)

    # VQ-VAE codebook
    p.add_argument("--num-embeddings", type=int, default=512)

    # WVQ per-subband codebook sizes
    p.add_argument("--cb-ll2", type=int, default=512)
    p.add_argument("--cb-lh2", type=int, default=128)
    p.add_argument("--cb-hl2", type=int, default=128)
    p.add_argument("--cb-hh2", type=int, default=32)
    p.add_argument("--cb-lh1", type=int, default=256)
    p.add_argument("--cb-hl1", type=int, default=256)
    p.add_argument("--cb-hh1", type=int, default=64)

    main(p.parse_args())
