# Angus claw-surface segmentation: ConvNeXt–U-Net and benchmark

Code for the paper *From medical imaging to the farm lane: a ConvNeXt–U-Net outperforms detection-based models and
matches DeepLabV3+ and SegFormer in segmenting Angus claws from video* (submitted to Livestock Science).

The repository contains everything needed to reproduce the models, the benchmark and the statistics of the paper:

| Part | Files |
|---|---|
| ConvNeXt–U-Net (three classes: background, claw, interdigital gap), label construction, inference with the watershed claw split | `clawseg/segmentation/unet*.py`, `clawseg/segmentation/pixelmap.py` (`split_claws`) |
| Training of the video-selected U-Nets | `scripts/train_unet.py` |
| Training of the primary U-Net and its variants | `scripts/train_variants.py` |
| Shared mask selection (centre in box, mask NMS, two largest masks) | `clawseg/segmentation/selection.py`, `scripts/benchmark.py` (`select`) |
| Metrics (IoU, Dice, Boundary IoU, HD95, claw-instance IoU, gap IoU, separation), box shift and midline displacement | `scripts/benchmark.py` |
| Benchmark adapters: YOLO26s-seg, RF-DETR-Seg, SAM 2.1 (zero-shot and fine-tuned), SAM 3, nnU-Net | `scripts/benchmark.py`, `clawseg/segmentation/{yolo,pixelmap}.py` |
| Fine-tuning of SAM 2.1 | `scripts/finetune_sam2.py`, `scripts/sam2_common.py` |
| YOLO training settings (claw segmenter, hoof detector) | `configs/*.yaml` |
| nnU-Net dataset export and Colab notebook | `scripts/nnunet_prepare.py`, `notebooks/nnunet_colab.ipynb` |
| Data-matched retraining of YOLO26s-seg and SAM 2.1 on the training set of the primary U-Net (Colab) | `scripts/make_matched_package.py`, `notebooks/matched_baselines_colab.ipynb` |
| Decoder controls DeepLabV3+ and SegFormer-B2 with the recipe of the primary U-Net (Colab) | `scripts/train_variants.py --arch`, `scripts/make_arch_package.py`, `notebooks/arch_controls_colab.ipynb` |
| Controlled downscaling and end-to-end evaluation with the hoof detector | `scripts/downscale_and_detector.py` |
| Statistics (bootstrap and frame-clustered bootstrap, Wilcoxon tests with Holm correction) | `scripts/stats_internal.py`, `scripts/stats_variants.py`, `scripts/stats_external.py` |

## Data and weights

Only the code is released. The images and annotations are the property of Te Mania Angus, and neither they nor the
trained weights are distributed. To run the scripts on your own data, provide two datasets in the YOLO-seg polygon format of a Roboflow export (`<split>/images`, `<split>/labels`, one class,
one polygon per claw):

```
data/ClawSet/{train,valid,test}/{images,labels}   # main dataset: 124 / 15 / 11 images, the test split is the internal test set
data/ClawTest/test/{images,labels}                # external test set: 32 frames of 29 bulls
```

Trained or downloaded weights are expected under `weights/` (SAM 2.1 and SAM 3 checkpoints are available from Meta):

```
weights/UNET-claw/unet_cnx.pt, unet_resnet34.pt          # video-selected U-Nets (scripts/train_unet.py)
weights/ablation/cnx_<variant>_s<seed>.pt                 # primary U-Net and variants (scripts/train_variants.py)
weights/ablation/{dlv3p_cnx,segformer_b2}_s<seed>.pt      # decoder controls (scripts/train_variants.py --arch)
weights/YOLO-SEG-claw/yolo26s-clawseg2.pt                 # configs/yolo26s_claw_seg.yaml
weights/YOLO-DET-hoof/yolo26m_v4-best.pt                  # configs/yolo26m_hoof_det.yaml
weights/RFDETR-SEG-claw/rfdetrmedium.pt                   # RF-DETR-Seg Medium (see below)
weights/SAM2.1-base/sam2.1_hiera_base_plus.pt             # official SAM 2.1 checkpoint
weights/SAM2.1-clawseg-lora/best.pt                       # scripts/finetune_sam2.py
weights/SAM3/sam3.pt                                      # official SAM 3 checkpoint
```

All locations can be changed with the environment variables `CLAWSEG_DATA`, `CLAWSEG_EXTERNAL`, `CLAWSEG_WEIGHTS`,
`CLAWSEG_RESULTS` and `CLAWSEG_NNUNET` (see `scripts/paths.py`).

## Installation

Tested with Python 3.13 and an NVIDIA RTX 3060 Laptop GPU (6 GB).

```
pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam2   # SAM 2.1 code, used by finetune_sam2.py and the SAM 2.1 adapter
```

## Reproducing the paper

`reproduce.sh` runs all steps in order. The individual steps are:

1. **Video-selected U-Nets** (Sections 2.2 and 2.3): `python scripts/train_unet.py --encoder convnext_tiny --size 320 --name unet_cnx`
   and `--encoder resnet34 --size 256 --name unet_resnet34` (80 epochs, checkpoint selection on all 15 validation images).
2. **Primary U-Net and variants** (Sections 2.3 and 2.7), each with seeds 0, 1 and 2 and 40 epochs:

   | Variant (Table 3) | Flags of `scripts/train_variants.py` |
   |---|---|
   | photos + video, three classes | `--encoder convnext_tiny --size 320 --epochs 40` |
   | photos + video, two classes | as above plus `--no_gap` |
   | photos, full recipe | `--encoder convnext_tiny --size 320 --epochs 40 --val_photos_only --exclude_train_source 20261007_134710-003` |
   | photos, flip only | as above plus `--flip_only 0.3` |
   | primary model (photos, enlarged windows) | as "photos, full recipe" plus `--box_scale 1.6` |
   | primary configuration, random encoder | as the primary model plus `--no_pretrained` |
   | DeepLabV3+ (decoder control) | as the primary model plus `--arch deeplabv3plus` (same ConvNeXt-Tiny encoder) |
   | SegFormer-B2 (decoder control) | as the primary model with `--arch segformer --encoder mit_b2` |

   The decoder controls were trained on Colab: `python scripts/make_arch_package.py`, run
   `notebooks/arch_controls_colab.ipynb` and unpack its results into `weights/ablation/`.

3. **nnU-Net** (Section 2.5): `python scripts/nnunet_prepare.py`, then `nnUNetv2_plan_and_preprocess -d 501 -c 2d` and
   `nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_100epochs` (four runs with different random initialisation; the
   notebook `notebooks/nnunet_colab.ipynb` runs three of them on Google Colab).
4. **SAM 2.1 fine-tuning** (Section 2.4): `python scripts/finetune_sam2.py --data data/ClawSet_coco --model base_plus
   --train-blocks 4 --epochs 20 --out weights/SAM2.1-clawseg-lora` (COCO export of the photograph training images).
5. **YOLO26s-seg and hoof detector**: `yolo train cfg=configs/yolo26s_claw_seg.yaml data=...` and
   `yolo train cfg=configs/yolo26m_hoof_det.yaml data=...`.
   Data-matched baselines: `python scripts/make_matched_package.py`, run `notebooks/matched_baselines_colab.ipynb`
   on Colab and unpack its results into `weights/matched/` (configurations `yolo_roi_matched`, `sam_ftm_*`).
6. **RF-DETR-Seg Medium** was trained on the Roboflow platform from the COCO-pre-trained checkpoint: 65 epochs,
   432 x 432 px, automatic orientation, letterboxing with black borders, horizontal flips with probability 0.3 and
   otherwise default hyper-parameters. The exported checkpoint is loaded with `rfdetr.RFDETRSegMedium`.
7. **Benchmark** (Sections 2.6 to 2.10):

   ```
   python scripts/benchmark.py --out results/benchmark                         # internal test set, Table 2
   python scripts/benchmark.py --jitter --out results/benchmark                # box shift
   python scripts/benchmark.py --ablation --out results/ablation               # U-Net variants
   python scripts/downscale_and_detector.py resolution                         # controlled downscaling
   python scripts/downscale_and_detector.py detector                           # hoof-detector boxes
   python scripts/benchmark.py --data data/ClawTest --models <names> --out results/clawtest/bench   # external test set
   ```

8. **Statistics**: `python scripts/stats_internal.py` (Table 2, nnU-Net average), `python scripts/stats_variants.py` (Table 3 and the variant tests) and `python scripts/stats_external.py` (Table 4).

Configuration names in `scripts/benchmark.py`: `unet_cnx`, `unet_r34` (video-selected U-Nets), `abl_cnx_<variant>_s<seed>`
(primary U-Net and variants), `nnunet_nomirror` and `nnunetc:<trainer>:<fold>:nomirror` (nnU-Net runs), `yolo_roi`,
`yolo_roi15`, `yolo_full`, `rfdetr_roi`, `rfdetr_roi15`, `rfdetr_full` (detection-based models, window 0.5, window 1.5,
full frame), `yolo_roi_c010`, `yolo_roi_c005`, `yolo_roi_c001` and the same for RF-DETR (confidence thresholds),
`sam_zs_{box,halves,claws}` and `sam_lora_{box,halves,claws}` (SAM 2.1 zero-shot and fine-tuned with whole-box,
half-box and two-stage prompts), `sam3_{text_full,text_roi,box,halves}` (SAM 3 zero-shot), `yolo_roi_matched`, `yolo_roi_matched_c001` and
`sam_ftm_{box,halves,claws}` (data-matched YOLO26s-seg and SAM 2.1).

## Verification

Run with the weights of the paper (not released), this code reproduces the per-hoof results of the paper exactly (checked for the primary
U-Net, the video-selected ConvNeXt–U-Net, YOLO26s-seg and the fine-tuned half-box SAM 2.1 on the internal test set),
and `scripts/stats_variants.py` reproduces Table 3 and the variant tests of Section 3.6.

## License and citation

MIT License (see `LICENSE`). If you use this code, please cite the paper:

> Later...
