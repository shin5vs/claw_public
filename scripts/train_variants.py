"""Training of the primary U-Net and its variants: the recipe of train_unet.py plus the switches below.

    --no_gap         two classes (background, claw): the interdigital gap is not a class, touching claws form one region
    --no_image_skip  no full-resolution input-image skip (only relevant for encoders whose first level has stride > 2)
    --val_photos_only        checkpoint selection on the photograph (IMG_*) validation images only, as for the
                             benchmark models, whose validation sets contain no video frame
    --exclude_train_source P drop training images whose source name starts with P (e.g. the test recording)
    --box_scale S            training windows are built from the claw box enlarged by a random factor in [1, S] and
                             shifted by up to 15% of its size, to cover the larger whole-hoof boxes of a hoof detector
    --no_pretrained          random encoder initialisation instead of the ImageNet weights
    --flip_only P            replace the augmentation by the recipe of RF-DETR-Seg: horizontal flip with probability P,
                             then resizing; no crop jitter, rotation, downscaling, JPEG or photometric changes

Data loading, augmentation, optimiser, schedule, mixed precision, validation and checkpoint selection are imported from
the original script unchanged; only the label construction (for --no_gap), the class count, the loss, the validation
score and the model's image skip differ. Example:

    # primary model (seeds 0, 1, 2)
    python scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 --val_photos_only \
        --exclude_train_source 20261007_134710-003 --box_scale 1.6 --seed 0 --name cnx_boxscale_s0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
spec = importlib.util.spec_from_file_location("train_unet", Path(__file__).resolve().parent / "train_unet.py")
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

from clawseg.segmentation.unet_data import CLAW, GAP, hoof_rois, label_map  # noqa: E402
from clawseg.segmentation.unet_model import ClawUNet, _ConvBlock  # noqa: E402

OUT = paths.WEIGHTS / "ablation"


def build_model(encoder: str, classes: int, image_skip: bool, pretrained: bool) -> ClawUNet:
    model = ClawUNet(encoder, classes=classes, pretrained=pretrained)
    if model.image_skip and not image_skip:
        model.image_skip = False
        model.full = _ConvBlock(model.full[0].in_channels - 3, 16)   # drop the 3 RGB input channels
    return model


def load_split(root: Path, split: str, no_gap: bool) -> list[dict]:
    """As T.load_split; with no_gap the label map is the union of the claw polygons (0 background, 1 claw)."""
    samples = []
    for img_path, image, polygons in T._images(root, split):
        if not polygons:
            continue
        h, w = image.shape[:2]
        labels, pairs, masks = label_map((h, w), polygons)
        if no_gap:
            labels = (np.logical_or.reduce(masks) * CLAW).astype(np.uint8)
        for x1, y1, x2, y2 in hoof_rois((h, w), masks, pairs):
            samples.append({"image": image[y1:y2, x1:x2].copy(), "labels": labels[y1:y2, x1:x2].copy(),
                            "source": img_path.name.split(".rf.")[0]})
    return samples


def load_split_wide(root: Path, split: str, no_gap: bool, pad: float = 2.0) -> list[dict]:
    """As load_split, but each sample keeps a wide crop (claw box enlarged by `pad` box sizes on each side) and the claw
    box inside it, so that windows of any scale up to `pad` can be cut at training time."""
    samples = []
    for img_path, image, polygons in T._images(root, split):
        if not polygons:
            continue
        h, w = image.shape[:2]
        labels, pairs, masks = label_map((h, w), polygons)
        if no_gap:
            labels = (np.logical_or.reduce(masks) * CLAW).astype(np.uint8)
        for x1, y1, x2, y2 in hoof_rois((h, w), masks, pairs, pad=0.0):
            bw, bh = x2 - x1, y2 - y1
            X1, Y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
            X2, Y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
            samples.append({"image": image[Y1:Y2, X1:X2].copy(), "labels": labels[Y1:Y2, X1:X2].copy(),
                            "box": (x1 - X1, y1 - Y1, x2 - X1, y2 - Y1), "source": img_path.name.split(".rf.")[0]})
    return samples


class ScaledBoxDataset(T.AugDataset):
    """T.AugDataset whose window comes from a randomly enlarged and shifted claw box (then the usual 0.5 padding)."""

    def __init__(self, samples, size, recipe, repeat, seed, max_scale):
        super().__init__(samples, size, recipe, repeat, seed)
        self.max_scale = max_scale

    def __getitem__(self, i):
        s = self.samples[i % len(self.samples)]
        x1, y1, x2, y2 = s["box"]
        bw, bh = x2 - x1, y2 - y1
        k = self.rng.uniform(1.0, self.max_scale)
        cx = (x1 + x2) / 2 + self.rng.uniform(-0.15, 0.15) * bw
        cy = (y1 + y2) / 2 + self.rng.uniform(-0.15, 0.15) * bh
        nw, nh = k * bw, k * bh
        H, W = s["image"].shape[:2]
        wx1, wy1 = max(0, int(cx - nw)), max(0, int(cy - nh))        # box of size k*bw padded by 0.5 on each side
        wx2, wy2 = min(W, int(cx + nw)), min(H, int(cy + nh))
        image, labels = T.augment(s["image"][wy1:wy2, wx1:wx2], s["labels"][wy1:wy2, wx1:wx2], self.rng, self.size, self.recipe)
        x = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - T.MEAN) / T.STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(labels.astype(np.int64))


def flip_only_augment(p_flip):
    def augment(image, labels, rng, size, recipe="base"):
        if rng.random() < p_flip:
            image, labels = image[:, ::-1], labels[:, ::-1]
        image = cv2.resize(np.ascontiguousarray(image), (size, size), interpolation=cv2.INTER_LINEAR)
        labels = cv2.resize(np.ascontiguousarray(labels), (size, size), interpolation=cv2.INTER_NEAREST)
        return image, labels
    return augment


def loss_fn(logits, target, class_weights, n_classes):
    """T.loss_fn generalised to 2 or 3 classes (Dice over claw, and gap when present)."""
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    onehot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    boundary = (F.max_pool2d(onehot, 3, 1, 1) + F.max_pool2d(-onehot, 3, 1, 1)).sum(1) > 0
    weights = 1.0 + 2.0 * boundary.float()
    ce = (ce * weights).sum() / weights.sum()
    probs = logits.float().softmax(1)
    fg = (CLAW, GAP) if n_classes == 3 else (CLAW,)
    dice = 0.0
    for c in fg:
        p, t = probs[:, c], (target == c).float()
        dice = dice + (1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    return ce + dice / 2 if n_classes == 3 else ce + 0.5 * dice


@torch.no_grad()
def evaluate(model, samples, size, device, n_classes, hoof_px=None):
    model.eval()
    inter, union = np.zeros(n_classes), np.zeros(n_classes)
    for k in range(0, len(samples), 16):
        batch = samples[k:k + 16]
        imgs = [cv2.resize(T.downscale(s["image"], hoof_px) if hoof_px else s["image"], (size, size),
                           interpolation=cv2.INTER_LINEAR) for s in batch]
        labs = np.stack([cv2.resize(s["labels"], (size, size), interpolation=cv2.INTER_NEAREST) for s in batch])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(T.to_tensor(imgs).to(device)).float().argmax(1).cpu().numpy()
        for c in range(n_classes):
            inter[c] += np.sum((pred == c) & (labs == c))
            union[c] += np.sum((pred == c) | (labs == c))
    iou = inter / np.maximum(union, 1)
    out = {"claw_iou": round(float(iou[CLAW]), 4)}
    if n_classes == 3:
        out["gap_iou"] = round(float(iou[GAP]), 4)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(paths.DATA))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--name", required=True)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-4)
    p.add_argument("--gap_weight", type=float, default=4.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_gap", action="store_true")
    p.add_argument("--no_image_skip", action="store_true")
    p.add_argument("--val_photos_only", action="store_true")
    p.add_argument("--exclude_train_source", default=None)
    p.add_argument("--flip_only", type=float, default=None)
    p.add_argument("--box_scale", type=float, default=None)
    p.add_argument("--no_pretrained", action="store_true", help="random encoder initialisation instead of ImageNet weights")
    args = p.parse_args(argv)
    if args.flip_only is not None:
        T.augment = flip_only_augment(args.flip_only)   # AugDataset looks the function up at call time

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_classes = 2 if args.no_gap else 3
    data = {split: load_split(root, split, args.no_gap) for split in ("train", "valid", "test")}
    sources = {split: {s["source"] for s in v} for split, v in data.items()}
    if sources["train"] & (sources["valid"] | sources["test"]):
        raise SystemExit("source image shared between training and validation/test")
    if args.val_photos_only:
        data["valid"] = [x for x in data["valid"] if x["source"].startswith("IMG_")]
    if args.exclude_train_source:
        data["train"] = [x for x in data["train"] if not x["source"].startswith(args.exclude_train_source)]
    if args.box_scale:
        data["train"] = [x for x in load_split_wide(root, "train", args.no_gap)
                         if not (args.exclude_train_source and x["source"].startswith(args.exclude_train_source))]
    train = list(data["train"])
    print({k: len(v) for k, v in data.items()}, "hoof regions", flush=True)

    model = build_model(args.encoder, n_classes, not args.no_image_skip, pretrained=not args.no_pretrained).to(device)
    enc = list(model.encoder.parameters())
    rest = [q for n, q in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_encoder}, {"params": rest, "lr": args.lr}],
                            weight_decay=args.weight_decay)
    dataset = (ScaledBoxDataset(train, args.size, "base", args.repeat, args.seed, args.box_scale) if args.box_scale
               else T.AugDataset(train, args.size, "base", args.repeat, args.seed))
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         worker_init_fn=T._seed_worker, persistent_workers=args.workers > 0)
    total, warmup = args.epochs * len(loader), 3 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    class_weights = torch.tensor([1.0, 1.0, args.gap_weight][:n_classes], device=device)
    classes = ["background", "claw", "gap"][:n_classes]
    config = {**vars(args), "n_classes": n_classes, "image_skip": model.image_skip,
              **{f"n_{k}": len(v) for k, v in data.items()}}
    (out / f"{args.name}_train_config.json").write_text(json.dumps(config, indent=2))

    best, best_score, t0 = None, -1.0, time.time()
    with (out / f"{args.name}_log.csv").open("w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "loss", "val_native", "val_video", "score", "sec"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(x)
                loss = loss_fn(logits.float(), y, class_weights, n_classes)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            native = evaluate(model, data["valid"], args.size, device, n_classes)
            video = evaluate(model, data["valid"], args.size, device, n_classes, T.VIDEO_HOOF_PX)
            score = float(np.mean(list(native.values()) + list(video.values())))   # same rule as the original script
            log.writerow([epoch, round(float(np.mean(losses)), 4), json.dumps(native), json.dumps(video),
                          round(score, 4), round(time.time() - t0)])
            fh.flush()
            if score > best_score:
                best_score = score
                best = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "encoder": args.encoder, "classes": classes, "input_size": args.size, "epoch": epoch,
                        "image_skip": model.image_skip, "valid": {"native": native, "video": video}}
                torch.save(best, out / f"{args.name}.pt")
            print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} native {native} video {video}", flush=True)
    metrics = {"best_epoch": best["epoch"], "valid": best["valid"], "train_seconds": round(time.time() - t0),
               "params_m": round(sum(q.numel() for q in model.parameters()) / 1e6, 2)}
    (out / f"{args.name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("best:", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
