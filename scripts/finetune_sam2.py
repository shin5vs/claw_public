#!/usr/bin/env python3
"""
Fine-tune SAM 2.1 (box prompt -> mask) on a small COCO-segmentation dataset of tiny objects (hooves).

What is trained: the mask decoder + prompt encoder (a few M params). The Hiera image encoder is frozen
by default (fits 6 GB easily); `--train-blocks N` also unfreezes the last N Hiera blocks + FPN neck
(lower LR) if the decoder alone is not enough.

Each training sample = one annotated instance: a random square crop (1.5-3x the box) around it, resized
to 1024, with a jittered box prompt (so it copes with imperfect YOLO/GroundingDINO boxes). Evaluation uses
the exact GT box and a fixed 2x crop, and is also run once BEFORE training (zero-shot baseline).

Dataset (COCO json with polygon/RLE `segmentation`), either
  --data DIR            DIR/train/ + DIR/valid/ (each: images + _annotations.coco.json, Roboflow layout)
  --images D --ann J    one json; validation is split by *group* (file stem minus trailing frame number,
                        see --group-regex) - neighbouring video frames must not land in both splits.

Fine-tuned SAM 2.1 of the paper (claw instances of the photographs in COCO format, 208 training instances):
    python scripts/finetune_sam2.py --data data/ClawSet_coco --model base_plus --train-blocks 4 --epochs 20 \
        --out weights/SAM2.1-clawseg-lora

Output: <out>/best.pt, <out>/last.pt (only the trainable tensors), <out>/train_config.json, <out>/log.csv
Use:    python scripts/benchmark.py --models sam_lora_box sam_lora_halves sam_lora_claws ...
"""
import argparse
import csv
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from sam2_common import (RES, VARIANTS, box_to_crop, build_model, crop_image, crop_window,  # noqa: E402
                         decode, encode, to_tensor)


# ----------------------------------------------------------------------------- data
def load_instances(img_dir: Path, ann_file: Path, min_area: float):
    coco = COCO(str(ann_file))
    items = []
    for ann in coco.dataset["annotations"]:
        if ann.get("iscrowd") or ann.get("area", 1) < min_area or not ann.get("segmentation"):
            continue
        info = coco.imgs[ann["image_id"]]
        path = img_dir / info["file_name"]
        if not path.exists():
            continue
        mask = coco.annToMask(ann)
        ys, xs = np.where(mask)
        if len(xs) < 4:
            continue
        items.append({"path": path, "ann": ann, "coco": coco,
                      "box": (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)})
    return items


def split_by_group(items, val_frac, group_regex, seed):
    groups = {}
    for it in items:
        g = re.sub(group_regex, "", it["path"].stem) or it["path"].stem
        groups.setdefault(g, []).append(it)
    names = sorted(groups)
    if len(names) < 2:
        raise SystemExit(f"--group-regex found only {len(names)} group(s); cannot split. Use --data with train/valid.")
    if len(names) > 0.8 * len({it["path"] for it in items}):
        print("[warn] almost every image is its own group -> split is effectively per-frame, "
              "val IoU will be optimistic. Check file names / --group-regex.")
    random.Random(seed).shuffle(names)
    target = max(1, round(val_frac * len(items)))
    val, train = [], []
    for g in names:
        (val if len(val) < target else train).extend(groups[g])
    print(f"[split] {len(names)} groups -> train {len(train)} instances, val {len(val)} instances")
    return train, val


class InstanceDS(Dataset):
    def __init__(self, items, train, repeat, scale_range, jitter, shift, flip=True):
        self.items, self.train, self.repeat = items, train, repeat if train else 1
        self.scale_range, self.jitter, self.shift, self.flip = scale_range, jitter, shift, flip
        self._cache = {}

    def __len__(self):
        return len(self.items) * self.repeat

    def _img(self, path):
        if path not in self._cache:
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    def __getitem__(self, i):
        it = self.items[i % len(self.items)]
        img = self._img(it["path"])
        mask = it["coco"].annToMask(it["ann"])
        box = np.array(it["box"], dtype=np.float32)
        if self.train:
            scale = random.uniform(*self.scale_range)
            shift = (random.uniform(-self.shift, self.shift), random.uniform(-self.shift, self.shift))
        else:
            scale, shift = 2.0, (0.0, 0.0)
        win = crop_window(box, scale, shift)
        im = crop_image(img, win)
        m = Image.fromarray(mask * 255).crop(
            (round(win[0]), round(win[1]), round(win[0] + win[2]), round(win[1] + win[2]))
        ).resize((RES, RES), Image.NEAREST)
        m = (np.asarray(m) > 127).astype(np.float32)
        b = box_to_crop(box, win)
        if self.train:
            wh = np.array([b[2] - b[0], b[3] - b[1]] * 2)
            b = b + np.random.randn(4) * self.jitter * wh
            b = np.clip(b, 0, RES - 1)
            if b[2] - b[0] < 4 or b[3] - b[1] < 4:
                b = box_to_crop(box, win)
            if self.flip and random.random() < 0.5:
                im, m = im.transpose(Image.FLIP_LEFT_RIGHT), m[:, ::-1].copy()
                b = np.array([RES - b[2], b[1], RES - b[0], b[3]], dtype=np.float32)
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.3))
            im = ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.3))
        return im, torch.from_numpy(m), torch.as_tensor(b, dtype=torch.float32)


def collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch])


# ----------------------------------------------------------------------------- loss / metric
def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - p_t) ** gamma
    loss = loss * (alpha * target + (1 - alpha) * (1 - target))
    return loss.flatten(1).mean(1)


def dice_loss(logits, target):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)


def iou_of(logits, target):
    pred = (logits > 0).flatten(1).float()
    t = target.flatten(1)
    inter = (pred * t).sum(1)
    return inter / (pred.sum(1) + t.sum(1) - inter).clamp(min=1)


# ----------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="dir with train/ and valid/ (COCO json beside the images)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--ann", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--group-regex", default=r"[_\-]?\d+$")
    ap.add_argument("--model", choices=list(VARIANTS), default="small")
    ap.add_argument("--ckpt", default=None, help="base checkpoint (default weights/SAM2.1-base/<variant file>)")
    ap.add_argument("--out", default="runs/sam2_hoof")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--repeat", type=int, default=4, help="passes over the instances per epoch (augmentation differs)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5, help="mask decoder / prompt encoder")
    ap.add_argument("--lr-encoder", type=float, default=5e-6, help="unfrozen Hiera blocks / neck")
    ap.add_argument("--train-blocks", type=int, default=0, help="also train the last N Hiera blocks + neck")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=20, help="optimizer steps")
    ap.add_argument("--min-scale", type=float, default=1.5, help="train crop = scale x box edge")
    ap.add_argument("--max-scale", type=float, default=3.0)
    ap.add_argument("--box-jitter", type=float, default=0.06, help="box coord noise, fraction of box size (std)")
    ap.add_argument("--shift", type=float, default=0.12, help="random crop-centre shift, fraction of crop side")
    ap.add_argument("--min-area", type=float, default=16.0, help="skip annotations smaller than this (px^2)")
    ap.add_argument("--w-focal", type=float, default=20.0)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument("--w-iou", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device
    amp = (not args.no_amp) and dev == "cuda"

    # ---- data
    if args.data:
        root = Path(args.data)
        vdir = next((root / n for n in ("valid", "val") if (root / n).is_dir()), None)
        train_items = load_instances(root / "train", root / "train" / "_annotations.coco.json", args.min_area)
        val_items = (load_instances(vdir, vdir / "_annotations.coco.json", args.min_area) if vdir else [])
        if not val_items:
            raise SystemExit("no validation split found (need DIR/valid/_annotations.coco.json)")
    elif args.images and args.ann:
        items = load_instances(Path(args.images), Path(args.ann), args.min_area)
        train_items, val_items = split_by_group(items, args.val_frac, args.group_regex, args.seed)
    else:
        raise SystemExit("give --data DIR, or --images DIR --ann FILE")
    print(f"[data] train {len(train_items)} instances / {len({i['path'] for i in train_items})} images, "
          f"val {len(val_items)} instances / {len({i['path'] for i in val_items})} images")
    if not train_items or not val_items:
        raise SystemExit("empty train or val set")

    tr_ds = InstanceDS(train_items, True, args.repeat, (args.min_scale, args.max_scale), args.box_jitter, args.shift)
    va_ds = InstanceDS(val_items, False, 1, (2.0, 2.0), 0, 0)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate, drop_last=len(tr_ds) > args.batch_size, persistent_workers=args.workers > 0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # ---- model
    model = build_model(args.model, args.ckpt, dev)
    for p in model.parameters():
        p.requires_grad = False
    # conv_s0/conv_s1 run inside forward_image (i.e. under no_grad while the encoder is frozen)
    dec_mods = [model.sam_mask_decoder, model.sam_prompt_encoder]
    skip = set() if args.train_blocks > 0 else {id(p) for m in (model.sam_mask_decoder.conv_s0, model.sam_mask_decoder.conv_s1)
                                                for p in m.parameters()}
    dec_params = [p for m in dec_mods for p in m.parameters() if id(p) not in skip]
    enc_params = []
    if args.train_blocks > 0:
        blocks = model.image_encoder.trunk.blocks
        mods = list(blocks[-args.train_blocks:]) + [model.image_encoder.neck]
        enc_params = [p for m in mods for p in m.parameters()]
    for p in dec_params + enc_params:
        p.requires_grad = True
    train_enc = bool(enc_params)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    n_tr = sum(p.numel() for p in dec_params + enc_params)
    print(f"[model] sam2.1 {args.model}: trainable {n_tr/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.1f}M"
          f" (encoder blocks: {args.train_blocks})")

    groups = [{"params": dec_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total_steps = max(1, args.epochs * len(tr_dl))

    def set_lr(step):
        f = (step + 1) / args.warmup if step < args.warmup else \
            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, total_steps - args.warmup)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    def forward(imgs, boxes, grad):
        x = to_tensor(imgs, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            emb, hr = encode(model, x, grad=grad and train_enc)
            return decode(model, emb, hr, boxes.to(dev))

    @torch.no_grad()
    def evaluate():
        model.eval()
        ious = []
        for imgs, masks, boxes in va_dl:
            logits, _ = forward(imgs, boxes, grad=False)
            ious += iou_of(logits, masks.to(dev)).tolist()
        return float(np.mean(ious))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps({**vars(args), "n_train": len(train_items), "n_val": len(val_items)}, indent=2))
    logf = open(out / "log.csv", "w", newline="")
    log = csv.writer(logf)
    log.writerow(["epoch", "train_loss", "train_iou", "val_iou", "lr", "sec"])

    def save(tag, extra):
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}
        torch.save({"trainable": sd, "variant": args.model, **extra}, out / f"{tag}.pt")

    base_iou = evaluate()
    print(f"[eval] zero-shot baseline (exact GT box, 2x crop) mean IoU = {base_iou:.4f}")
    log.writerow([0, "", "", f"{base_iou:.4f}", "", 0])
    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not train_enc:
            model.image_encoder.eval()
        tl, ti, n = 0.0, 0.0, 0
        for imgs, masks, boxes in tr_dl:
            set_lr(step)
            masks = masks.to(dev)[:, None]
            logits, iou_pred = forward(imgs, boxes, grad=True)
            lf, ld = focal_loss(logits, masks).mean(), dice_loss(logits, masks).mean()
            true_iou = iou_of(logits.detach(), masks)
            li = torch.nn.functional.mse_loss(iou_pred.float(), true_iou)
            loss = args.w_focal * lf + args.w_dice * ld + args.w_iou * li
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params + enc_params, 1.0)
            opt.step()
            step += 1
            tl += loss.item() * len(imgs)
            ti += true_iou.sum().item()
            n += len(imgs)
        row = [epoch, f"{tl/n:.4f}", f"{ti/n:.4f}", "", f"{opt.param_groups[0]['lr']:.2e}", f"{time.time()-t0:.0f}"]
        msg = f"epoch {epoch:3d}/{args.epochs}  loss {tl/n:.4f}  train IoU {ti/n:.3f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            v = evaluate()
            row[3] = f"{v:.4f}"
            msg += f"  val IoU {v:.4f} (zero-shot {base_iou:.4f})"
            if v > best:
                best = v
                save("best", {"epoch": epoch, "val_iou": v})
                msg += "  * best"
        print(msg + f"  [{torch.cuda.max_memory_allocated()/2**30:.1f} GiB peak]" if dev == "cuda" else msg, flush=True)
        log.writerow(row)
        logf.flush()
    save("last", {"epoch": args.epochs})
    print(f"done. best val IoU {best:.4f} vs zero-shot {base_iou:.4f}  ->  {out}/best.pt")


if __name__ == "__main__":
    main()
