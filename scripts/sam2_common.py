"""Shared helpers for the SAM 2.1 box-prompt fine-tune / inference scripts.

Works with a clone of https://github.com/facebookresearch/sam2 in <repo>/sam2 (no pip install of sam2 needed) and
the official SAM 2.1 checkpoints in <repo>/weights/SAM2.1-base.

Design: the model gets a *crop* around the prompt box (the hoof is tiny in the full frame), resized
to 1024x1024, plus the box as prompt. Train and inference use the same crop geometry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# `import sam2` must resolve to <repo>/sam2/sam2 (the package), so put the clone's root on the path.
sys.path.insert(0, str(ROOT / "sam2"))

RES = 1024
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# variant -> (hydra config, checkpoint file name)
VARIANTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
DEFAULT_CKPT_DIR = ROOT / "weights" / "SAM2.1-base"


def build_model(variant: str, ckpt: str | None, device: str, finetuned: str | None = None):
    """Base SAM 2.1 model; optionally overlay a checkpoint written by finetune_sam2.py."""
    from sam2.build_sam import build_sam2

    cfg, fname = VARIANTS[variant]
    ckpt = ckpt or str(DEFAULT_CKPT_DIR / fname)
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}\ndownload {fname} from https://github.com/facebookresearch/sam2")
    # apply_postprocessing=False: no dynamic multimask fallback, so inference uses the very mask
    # token that is trained (single-mask output for a box prompt).
    model = build_sam2(cfg, ckpt, device=device, mode="eval", apply_postprocessing=False)
    if finetuned:
        state = torch.load(finetuned, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state["trainable"], strict=False)
        if unexpected:
            raise SystemExit(f"unexpected keys in {finetuned}: {unexpected[:5]}")
    return model


def crop_window(box_xyxy, scale: float, shift=(0.0, 0.0), min_side: float = 64.0):
    """Square window (x0, y0, side) centred on the box, `scale` x its longer edge."""
    x0, y0, x1, y1 = box_xyxy
    side = max(max(x1 - x0, y1 - y0) * scale, min_side)
    cx = (x0 + x1) / 2 + shift[0] * side
    cy = (y0 + y1) / 2 + shift[1] * side
    return cx - side / 2, cy - side / 2, side


def crop_image(img: Image.Image, win) -> Image.Image:
    x0, y0, side = win
    box = (round(x0), round(y0), round(x0 + side), round(y0 + side))  # outside the image -> black
    return img.crop(box).resize((RES, RES), Image.BILINEAR)


def box_to_crop(box_xyxy, win) -> np.ndarray:
    x0, y0, side = win
    k = RES / side
    b = np.array(box_xyxy, dtype=np.float32)
    b[[0, 2]] = (b[[0, 2]] - x0) * k
    b[[1, 3]] = (b[[1, 3]] - y0) * k
    return b


def to_tensor(imgs: list[Image.Image], device) -> torch.Tensor:
    x = torch.from_numpy(np.stack([np.asarray(i.convert("RGB")) for i in imgs])).permute(0, 3, 1, 2).float() / 255
    return ((x - MEAN) / STD).to(device)


def encode(model, x: torch.Tensor, grad: bool = False):
    """Image embedding exactly as SAM2ImagePredictor.set_image computes it (batched)."""
    with torch.set_grad_enabled(grad):
        out = model.forward_image(x)
        _, feats, _, sizes = model._prepare_backbone_features(out)
        if model.directly_add_no_mem_embed:
            feats[-1] = feats[-1] + model.no_mem_embed
        feats = [f.permute(1, 2, 0).reshape(x.shape[0], -1, *s) for f, s in zip(feats[::-1], sizes[::-1])][::-1]
    return feats[-1], feats[:-1]


def decode(model, embed, high_res, boxes_crop: torch.Tensor):
    """Box prompt (B,4) in 1024 crop pixels -> mask logits (B,1,1024,1024) and predicted IoU (B,)."""
    B = boxes_crop.shape[0]
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes_crop.device).repeat(B, 1)
    sparse, dense = model.sam_prompt_encoder(points=(boxes_crop.reshape(-1, 2, 2), labels), boxes=None, masks=None)
    low, iou, _, _ = model.sam_mask_decoder(
        image_embeddings=embed,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res,
    )
    logits = F.interpolate(low.float(), size=(RES, RES), mode="bilinear", align_corners=False)
    return logits, iou.reshape(B)


@torch.no_grad()
def segment_boxes(model, img: Image.Image, boxes_xyxy, scale: float = 2.0, device="cuda", amp=True):
    """Full-frame bool masks (N,H,W) + predicted IoUs for each box prompt on `img`."""
    W, H = img.size
    masks, ious = [], []
    for box in boxes_xyxy:
        win = crop_window(box, scale)
        x = to_tensor([crop_image(img, win)], device)
        b = torch.from_numpy(box_to_crop(box, win)).to(device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            emb, hr = encode(model, x)
            logits, iou = decode(model, emb, hr, b)
        masks.append(paste_back(logits[0, 0].cpu().numpy() > 0, win, (H, W)))
        ious.append(float(iou[0]))
    return (np.stack(masks) if masks else np.zeros((0, H, W), bool)), ious


def paste_back(mask_crop: np.ndarray, win, hw) -> np.ndarray:
    """Resize a 1024x1024 crop-space mask to the window and paste it into an HxW canvas."""
    import cv2

    H, W = hw
    x0, y0, side = win
    xi, yi, s = round(x0), round(y0), max(round(side), 1)
    m = cv2.resize(mask_crop.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W), np.uint8)
    sx0, sy0 = max(xi, 0), max(yi, 0)
    sx1, sy1 = min(xi + s, W), min(yi + s, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0:sy1, sx0:sx1] = m[sy0 - yi:sy1 - yi, sx0 - xi:sx1 - xi]
    return out.astype(bool)
