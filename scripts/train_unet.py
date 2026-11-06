"""Training of the three-class ConvNeXt-U-Net (background / claw / interdigital gap) on hoof ROIs from YOLO polygons.

Video-selected models of the paper (checkpoint selection on all 15 validation images, 80 epochs):

    python scripts/train_unet.py --encoder convnext_tiny --size 320 --name unet_cnx
    python scripts/train_unet.py --encoder resnet34 --size 256 --name unet_resnet34

The train / valid / test split of the dataset export is kept (no source image is shared between subsets). Validation
runs at two scales: native (the ROI resized to the input size) and video-like (the ROI first downscaled so that the
hoof is 30 px high, then resized back), because hooves in video frames are 12-35 px high. The best checkpoint maximises
the mean of the four IoUs (claw and gap, native and video-like). Augmentation (`--aug`): `base` (used in the paper) =
crop jitter, flip, rotation, brightness, contrast, gamma, blur, noise and, with probability 0.6, downscaling of the
hoof to 12-60 px with JPEG compression. `video` and `strong` are stronger recipes not used in the paper. `--bg_ratio`:
share of claw-free background crops. `--ema`: an exponential moving average of the weights is evaluated and saved.
Output: `<out>/<name>.pt`, `<name>_log.csv`, `<name>_metrics.json`.
"""
from __future__ import annotations

import argparse
import csv
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
import paths  # noqa: E402  (makes clawseg importable)

from clawseg.segmentation.unet_data import CLASS_NAMES, CLAW, GAP, hoof_rois, label_map, read_yolo_polygons  # noqa: E402
from clawseg.segmentation.unet_model import IMAGENET_MEAN, IMAGENET_STD, ClawUNet  # noqa: E402

MEAN, STD = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
VIDEO_HOOF_PX = 30          # hoof height of the video-like validation


def _images(root: Path, split: str):
    for img_path in sorted((root / split / "images").glob("*")):
        label_path = root / split / "labels" / (img_path.stem + ".txt")
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        yield img_path, image, (read_yolo_polygons(label_path, w, h) if label_path.exists() else [])


def load_split(root: Path, split: str) -> list[dict]:
    samples = []
    for img_path, image, polygons in _images(root, split):
        if not polygons:
            continue
        h, w = image.shape[:2]
        labels, pairs, masks = label_map((h, w), polygons)
        for x1, y1, x2, y2 in hoof_rois((h, w), masks, pairs):
            samples.append({"image": image[y1:y2, x1:x2].copy(), "labels": labels[y1:y2, x1:x2].copy(),
                            "source": img_path.name.split(".rf.")[0]})
    return samples


def load_background(root: Path, split: str, per_image: int, rng: random.Random) -> list[dict]:
    """Claw-free crops (claw pixels < 1%) from the training images, of hoof-ROI size."""
    out = []
    for img_path, image, polygons in _images(root, split):
        h, w = image.shape[:2]
        labels = label_map((h, w), polygons)[0] if polygons else np.zeros((h, w), np.uint8)
        for _ in range(per_image * 10):
            if sum(s["source"] == img_path.name for s in out) >= per_image:
                break
            side = int(min(h, w) * rng.uniform(0.15, 0.5))
            if side < 16:
                break
            x, y = rng.randint(0, w - side), rng.randint(0, h - side)
            if np.count_nonzero(labels[y:y + side, x:x + side]) < 0.01 * side * side:
                out.append({"image": image[y:y + side, x:x + side].copy(), "labels": np.zeros((side, side), np.uint8),
                            "source": img_path.name})
    return out


# ---------------------------------------------------------------------- augmentation

def downscale(image: np.ndarray, hoof_px: float) -> np.ndarray:
    """Downscales the ROI so that the hoof (half of the ROI height at a margin of 0.5) is `hoof_px` high."""
    h, w = image.shape[:2]
    f = 2 * hoof_px / h
    if f >= 1:
        return image
    return cv2.resize(image, (max(4, round(w * f)), max(4, round(h * f))), interpolation=cv2.INTER_AREA)


def augment(image: np.ndarray, labels: np.ndarray, rng: random.Random, size: int, recipe: str = "base"):
    video, strong = recipe == "video", recipe == "strong"
    h, w = image.shape[:2]
    # ROI jitter: detector boxes are not exact either (0-12% cropped on each side)
    x1, x2 = int(w * rng.uniform(0, 0.12)), w - int(w * rng.uniform(0, 0.12))
    y1, y2 = int(h * rng.uniform(0, 0.12)), h - int(h * rng.uniform(0, 0.12))
    image, labels = image[y1:y2, x1:x2], labels[y1:y2, x1:x2]
    if rng.random() < 0.5:
        image, labels = image[:, ::-1], labels[:, ::-1]
    if rng.random() < (0.7 if strong else 0.5):
        hh, ww = image.shape[:2]
        angle, zoom = (rng.uniform(-25, 25), rng.uniform(0.8, 1.25)) if strong else (rng.uniform(-15, 15), 1.0)
        rot = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, zoom)
        image = cv2.warpAffine(np.ascontiguousarray(image), rot, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        border = dict(borderMode=cv2.BORDER_REFLECT) if strong else dict(borderValue=0)   # with reflected borders the mirrored edge is labelled too
        labels = cv2.warpAffine(np.ascontiguousarray(labels), rot, (ww, hh), flags=cv2.INTER_NEAREST, **border)
    if strong and rng.random() < 0.4:    # perspective: camera height and angle vary between hooves
        hh, ww = image.shape[:2]
        src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
        dst = src + np.float32([[rng.uniform(-0.12, 0.12) * ww, rng.uniform(-0.12, 0.12) * hh] for _ in range(4)])
        warp = cv2.getPerspectiveTransform(src, dst)
        image = cv2.warpPerspective(np.ascontiguousarray(image), warp, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        labels = cv2.warpPerspective(np.ascontiguousarray(labels), warp, (ww, hh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if rng.random() < (0.8 if video else 0.6):   # scale: pixelation and blur of video hooves
        image = downscale(image, rng.uniform(10, 45) if video else rng.uniform(12, 60))
        if rng.random() < (0.7 if video else 0.5):
            q = rng.randint(30, 80) if video else rng.randint(40, 90)
            image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    image = cv2.resize(np.ascontiguousarray(image), (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    labels = cv2.resize(np.ascontiguousarray(labels), (size, size), interpolation=cv2.INTER_NEAREST)
    if strong and rng.random() < 0.5:    # colour: hue and saturation shift
        hsv = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.4), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    image = image * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)                       # contrast, brightness
    image = 255.0 * (np.clip(image, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)          # gamma
    if video and rng.random() < 0.5:     # dark horn on dark ground: darkening and contrast towards the mean
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (mean + (image - mean) * rng.uniform(0.5, 0.9)) * rng.uniform(0.35, 0.8)
    if video and rng.random() < 0.3:     # faded colours
        gray = image.mean(axis=2, keepdims=True)
        image = gray + (image - gray) * rng.uniform(0.2, 0.7)
    if rng.random() < (0.4 if video else 0.3):
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 2.0 if video else 1.5))
    if video and rng.random() < 0.2:     # motion blur (the bull is walking)
        k = rng.choice((3, 5, 7))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2] = 1.0 / k
        image = cv2.filter2D(image, -1, kernel if rng.random() < 0.5 else kernel.T)
    if rng.random() < 0.3:
        image = image + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 8), image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), labels


def to_tensor(images: list[np.ndarray]) -> torch.Tensor:
    x = np.stack([(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD for im in images])
    return torch.from_numpy(x.transpose(0, 3, 1, 2))


class AugDataset(torch.utils.data.Dataset):
    """The training ROIs `repeat` times per epoch with random augmentation (separate random stream per worker)."""

    def __init__(self, samples: list[dict], size: int, recipe: str, repeat: int, seed: int) -> None:
        self.samples, self.size, self.recipe, self.repeat = samples, size, recipe, repeat
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, i: int):
        s = self.samples[i % len(self.samples)]
        image, labels = augment(s["image"], s["labels"], self.rng, self.size, self.recipe)
        x = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(labels.astype(np.int64))


def _seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(torch.initial_seed() % 2**32)


# ---------------------------------------------------------------------- loss and metrics

def loss_fn(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Cross-entropy (weighted by class and boundary) plus the Dice losses of claw and gap."""
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
    onehot = F.one_hot(target, len(CLASS_NAMES)).permute(0, 3, 1, 2).float()
    boundary = (F.max_pool2d(onehot, 3, 1, 1) + F.max_pool2d(-onehot, 3, 1, 1)).sum(1) > 0
    weights = 1.0 + 2.0 * boundary.float()
    ce = (ce * weights).sum() / weights.sum()
    probs = logits.float().softmax(1)
    dice = 0.0
    for c in (CLAW, GAP):
        p, t = probs[:, c], (target == c).float()
        dice = dice + (1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()
    return ce + dice / 2


@torch.no_grad()
def evaluate(model, samples: list[dict], size: int, device, hoof_px: float | None = None, tta: bool = False) -> dict:
    """IoU per class over the whole set (pixel sums). `hoof_px`: hoof first downscaled to this height.
    `tta`: average of the probabilities of the image and its mirror image."""
    model.eval()
    inter, union = np.zeros(3), np.zeros(3)
    for k in range(0, len(samples), 16):
        batch = samples[k:k + 16]
        imgs = [cv2.resize(downscale(s["image"], hoof_px) if hoof_px else s["image"], (size, size),
                           interpolation=cv2.INTER_LINEAR) for s in batch]
        labs = np.stack([cv2.resize(s["labels"], (size, size), interpolation=cv2.INTER_NEAREST) for s in batch])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            x = to_tensor(imgs).to(device)
            probs = model(x).float().softmax(1)
            if tta:
                probs = (probs + model(x.flip(-1)).float().softmax(1).flip(-1)) / 2
            pred = probs.argmax(1).cpu().numpy()
        for c in range(3):
            inter[c] += np.sum((pred == c) & (labs == c))
            union[c] += np.sum((pred == c) | (labs == c))
    iou = inter / np.maximum(union, 1)
    return {"claw_iou": round(float(iou[CLAW]), 4), "gap_iou": round(float(iou[GAP]), 4)}


# ---------------------------------------------------------------------- training

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=str(paths.DATA))
    p.add_argument("--out", default=str(paths.WEIGHTS / "UNET-claw"))
    p.add_argument("--name", default=None, help="Name of the output files (default: unet_<encoder>)")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--repeat", type=int, default=3, help="Passes over the training ROIs per epoch")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-4)
    p.add_argument("--gap_weight", type=float, default=4.0, help="Weight of the gap class in the cross-entropy")
    p.add_argument("--aug", choices=("base", "video", "strong"), default="base")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema", type=float, default=0.0, help="Exponential moving average of the weights (e.g. 0.995, 0 = off), evaluated and saved")
    p.add_argument("--bg_ratio", type=float, default=0.0, help="Share of claw-free background crops relative to the training ROIs")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Deterministic cuDNN/cuBLAS (slower), non-deterministic operations raise warnings")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    name = args.name or f"unet_{args.encoder}"

    if args.deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {split: load_split(root, split) for split in ("train", "valid", "test")}
    sources = {split: {s["source"] for s in v} for split, v in data.items()}
    if sources["train"] & (sources["valid"] | sources["test"]):
        raise SystemExit("the same source image occurs in the training and the validation or test set")
    train = list(data["train"])
    if args.bg_ratio > 0:
        background = load_background(root, "train", per_image=2, rng=rng)
        rng.shuffle(background)
        train += background[: int(args.bg_ratio * len(data["train"]))]
    print({k: len(v) for k, v in data.items()}, "hoof ROIs; training samples including background crops:", len(train), flush=True)

    model = ClawUNet(args.encoder, classes=len(CLASS_NAMES), pretrained=True).to(device)
    enc = list(model.encoder.parameters())
    rest = [q for n, q in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_encoder}, {"params": rest, "lr": args.lr}], weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(AugDataset(train, args.size, args.aug, args.repeat, args.seed), batch_size=args.batch,
                                         shuffle=True, num_workers=args.workers, worker_init_fn=_seed_worker,
                                         persistent_workers=args.workers > 0, drop_last=False)
    total, warmup = args.epochs * len(loader), 3 * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    class_weights = torch.tensor([1.0, 1.0, args.gap_weight], device=device)
    ema = (torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema),
                                               use_buffers=True) if args.ema > 0 else None)
    scored = ema.module if ema is not None else model    # this model is evaluated and saved

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out / f"{name}_train_config.json").write_text(json.dumps({**config, **{f"n_{k}": len(v) for k, v in data.items()},
                                                              "n_train_samples": len(train)}, indent=2))
    best, best_score = None, -1.0
    with (out / f"{name}_log.csv").open("w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["epoch", "loss", "val_claw", "val_gap", "val_claw_video", "val_gap_video", "sec"])
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(x)
                loss = loss_fn(logits.float(), y, class_weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                if ema is not None:
                    ema.update_parameters(model)
                losses.append(loss.item())
            native = evaluate(scored, data["valid"], args.size, device)
            video = evaluate(scored, data["valid"], args.size, device, VIDEO_HOOF_PX)
            score = np.mean([native["claw_iou"], native["gap_iou"], video["claw_iou"], video["gap_iou"]])
            log.writerow([epoch, round(float(np.mean(losses)), 4), native["claw_iou"], native["gap_iou"],
                          video["claw_iou"], video["gap_iou"], round(time.time() - t0)])
            fh.flush()
            mark = ""
            if score > best_score:
                best_score, mark = score, " *"
                best = {"state_dict": {k: v.detach().cpu() for k, v in scored.state_dict().items()},
                        "encoder": args.encoder, "classes": list(CLASS_NAMES), "input_size": args.size, "epoch": epoch,
                        "valid": {"native": native, "video": video}}
                torch.save(best, out / f"{name}.pt")
            print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} | valid native {native} | video-like {video}{mark}", flush=True)

    scored.load_state_dict(best["state_dict"])
    test = {"native": evaluate(scored, data["test"], args.size, device),
            "video": evaluate(scored, data["test"], args.size, device, VIDEO_HOOF_PX)}
    metrics = {"best_epoch": best["epoch"], "valid": best["valid"], "test": test,
               "params_m": round(sum(q.numel() for q in model.parameters()) / 1e6, 1), "train_seconds": round(time.time() - t0)}
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("best:", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
