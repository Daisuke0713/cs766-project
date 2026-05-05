#!/bin/bash
set -e

EPOCHS=${1:-200}
GPU=${2:-0}

# cifar10: vqvae baseline (629K params)
python train.py --model vqvae --output out/vqvae \
    --epochs $EPOCHS --gpu $GPU --dataset cifar10 \
    --hidden 128 --res-hidden 32 --embed-dim 64 --num-embeddings 512

# cifar10: wvq (922K params)
python train.py --model wvq --output out/wvq \
    --epochs $EPOCHS --gpu $GPU --dataset cifar10 \
    --hidden 64 --res-hidden 32 --embed-dim 32 \
    --cb-ll2 512 --cb-lh2 128 --cb-hl2 128 --cb-hh2 32 \
    --cb-lh1 256 --cb-hl1 256 --cb-hh1 64

# cifar10: vqvae param-equalized (~906K params, hidden=160)
python train.py --model vqvae --output out/vqvae_eq \
    --epochs $EPOCHS --gpu $GPU --dataset cifar10 \
    --hidden 160 --res-hidden 32 --embed-dim 64 --num-embeddings 512

# cifar10: wvq codebook reallocation ablation
python train.py --model wvq --output out/wvq_realloc \
    --epochs $EPOCHS --gpu $GPU --dataset cifar10 \
    --hidden 64 --res-hidden 32 --embed-dim 32 \
    --cb-ll2 256 --cb-lh2 256 --cb-hl2 256 --cb-hh2 32 \
    --cb-lh1 512 --cb-hl1 512 --cb-hh1 64

# cifar100
python train.py --model vqvae --output out/vqvae_cifar100 \
    --epochs $EPOCHS --gpu $GPU --dataset cifar100 \
    --hidden 160 --res-hidden 32 --embed-dim 64 --num-embeddings 512

python train.py --model wvq --output out/wvq_cifar100 \
    --epochs $EPOCHS --gpu $GPU --dataset cifar100 \
    --hidden 64 --res-hidden 32 --embed-dim 32 \
    --cb-ll2 512 --cb-lh2 128 --cb-hl2 128 --cb-hh2 32 \
    --cb-lh1 256 --cb-hl1 256 --cb-hh1 64

# stl10 at 64x64
python train.py --model vqvae --output out/vqvae_stl10 \
    --epochs $EPOCHS --gpu $GPU --dataset stl10 --image-size 64 \
    --hidden 160 --res-hidden 32 --embed-dim 64 --num-embeddings 512

python train.py --model wvq --output out/wvq_stl10 \
    --epochs $EPOCHS --gpu $GPU --dataset stl10 --image-size 64 \
    --hidden 64 --res-hidden 32 --embed-dim 32 \
    --cb-ll2 512 --cb-lh2 128 --cb-hl2 128 --cb-hh2 32 \
    --cb-lh1 256 --cb-hl1 256 --cb-hh1 64

# reports
python evaluate.py --vqvae-dir out/vqvae_eq --wvq-dir out/wvq --output report_equalized/ --dataset cifar10 --gpu $GPU
python evaluate.py --vqvae-dir out/wvq --wvq-dir out/wvq_realloc --output report_realloc/ --dataset cifar10 --gpu $GPU
python evaluate.py --vqvae-dir out/vqvae_cifar100 --wvq-dir out/wvq_cifar100 --output report_cifar100/ --dataset cifar100 --gpu $GPU
python evaluate.py --vqvae-dir out/vqvae_stl10 --wvq-dir out/wvq_stl10 --output report_stl10/ --dataset stl10 --image-size 64 --gpu $GPU
