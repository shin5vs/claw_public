#!/usr/bin/env bash
# Reproduces the models, benchmark and statistics of the paper. Expects data/ and weights/ as described in README.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

# 1. video-selected U-Nets (80 epochs)
$PY scripts/train_unet.py --encoder convnext_tiny --size 320 --name unet_cnx
$PY scripts/train_unet.py --encoder resnet34 --size 256 --name unet_resnet34

# 2. primary U-Net and variants (40 epochs, seeds 0-2)
PHOTOS="--val_photos_only --exclude_train_source 20261007_134710-003"
for s in 0 1 2; do
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 --seed $s --name cnx_full_s$s
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 --no_gap --seed $s --name cnx_nogap_s$s
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 $PHOTOS --seed $s --name cnx_photoval_s$s
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 $PHOTOS --flip_only 0.3 --seed $s --name cnx_fliponly_s$s
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 $PHOTOS --box_scale 1.6 --seed $s --name cnx_boxscale_s$s
  $PY scripts/train_variants.py --encoder convnext_tiny --size 320 --epochs 40 $PHOTOS --box_scale 1.6 --no_pretrained --seed $s --name cnx_scratch_s$s
  $PY scripts/train_variants.py --arch deeplabv3plus --encoder convnext_tiny --size 320 --epochs 40 $PHOTOS --box_scale 1.6 --seed $s --name dlv3p_cnx_s$s
  $PY scripts/train_variants.py --arch segformer --encoder mit_b2 --size 320 --epochs 40 $PHOTOS --box_scale 1.6 --seed $s --name segformer_b2_s$s
done

# 3. nnU-Net (four 100-epoch runs, see notebooks/nnunet_colab.ipynb for runs on Colab)
$PY scripts/nnunet_prepare.py
export nnUNet_raw=nnunet/raw nnUNet_preprocessed=nnunet/preprocessed nnUNet_results=nnunet/results
nnUNetv2_plan_and_preprocess -d 501 -c 2d
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_100epochs

# 4.-6. SAM 2.1 fine-tuning, YOLO models (RF-DETR-Seg was trained on the Roboflow platform, see README.md)
$PY scripts/finetune_sam2.py --data data/ClawSet_coco --model base_plus --train-blocks 4 --epochs 20 --out weights/SAM2.1-clawseg-lora
yolo train cfg=configs/yolo26s_claw_seg.yaml data=data/claw_crops/data.yaml
yolo train cfg=configs/yolo26m_hoof_det.yaml data=data/hoof_boxes/data.yaml

# data-matched baselines: build the Colab package, run notebooks/matched_baselines_colab.ipynb, unpack into weights/matched/
$PY scripts/make_matched_package.py

# 7. benchmark on the internal test set
CMP="yolo_roi rfdetr_roi sam_zs_halves sam_lora_halves sam3_text_full sam3_halves"
PRIMARY="abl_cnx_boxscale_s0 abl_cnx_boxscale_s1 abl_cnx_boxscale_s2"
NNC="nnunetc:nnUNetTrainer_100epochs:0:nomirror nnunetc:nnUNetTrainer_100epochs:1:nomirror nnunetc:nnUNetTrainer_100epochs:2:nomirror"
$PY scripts/benchmark.py --out results/benchmark                                  # all configurations of Table 2
$PY scripts/benchmark.py --jitter --models $CMP --out results/benchmark
$PY scripts/benchmark.py --ablation --out results/ablation --repeats 1             # primary U-Net and all variants
$PY scripts/benchmark.py --ablation --jitter --out results/ablation
$PY scripts/benchmark.py --models nnunet_nomirror --out results/nnunet_nomirror    # first nnU-Net run (local)
$PY scripts/benchmark.py --models nnunet_nomirror --jitter --out results/nnunet_nomirror
$PY scripts/benchmark.py --models $NNC --out results/nnunet_colab                  # three runs from nnunet/results_colab
$PY scripts/benchmark.py --models $NNC --jitter --out results/nnunet_colab
$PY scripts/downscale_and_detector.py resolution                                   # benchmark models
$PY scripts/downscale_and_detector.py resolution --models $PRIMARY abl_dlv3p_cnx_s0 abl_dlv3p_cnx_s1 abl_dlv3p_cnx_s2 abl_segformer_b2_s0 abl_segformer_b2_s1 abl_segformer_b2_s2 --out ablation/resolution_per_sample.csv
$PY scripts/downscale_and_detector.py resolution --models nnunet_nomirror --out nnunet_nomirror/resolution_per_sample.csv
$PY scripts/downscale_and_detector.py resolution --models $NNC --out nnunet_colab/resolution_per_sample.csv
$PY scripts/downscale_and_detector.py detector
$PY scripts/downscale_and_detector.py detector --models abl_cnx_full_s0 abl_cnx_full_s1 abl_cnx_full_s2 abl_cnx_nogap_s0 abl_cnx_nogap_s1 abl_cnx_nogap_s2 abl_cnx_photoval_s0 abl_cnx_photoval_s1 abl_cnx_photoval_s2 abl_cnx_fliponly_s0 abl_cnx_fliponly_s1 abl_cnx_fliponly_s2 $PRIMARY abl_cnx_scratch_s0 abl_cnx_scratch_s1 abl_cnx_scratch_s2 abl_dlv3p_cnx_s0 abl_dlv3p_cnx_s1 abl_dlv3p_cnx_s2 abl_segformer_b2_s0 abl_segformer_b2_s1 abl_segformer_b2_s2 --out ablation/detector_per_hoof.csv
$PY scripts/downscale_and_detector.py detector --models nnunet_nomirror --out nnunet_nomirror/detector_per_hoof.csv
$PY scripts/downscale_and_detector.py detector --models $NNC --out nnunet_colab/detector_per_hoof.csv
$PY scripts/benchmark.py --models yolo_roi_c010 yolo_roi_c005 yolo_roi_c001 rfdetr_roi_c010 rfdetr_roi_c005 rfdetr_roi_c001 --out results/lowconf_internal

# 8. external test set
EXT="--data ${CLAWSEG_EXTERNAL:-data/ClawTest}"
$PY scripts/benchmark.py $EXT --models unet_cnx unet_r34 yolo_roi rfdetr_roi sam_zs_box sam_zs_halves sam_lora_box sam_lora_halves sam3_text_full sam3_box sam3_halves nnunet_nomirror $NNC --out results/clawtest/bench --repeats 1
$PY scripts/benchmark.py $EXT --jitter --models $CMP nnunet_nomirror $NNC --out results/clawtest/bench
$PY scripts/benchmark.py $EXT --ablation --out results/clawtest/abl --repeats 1
CONTROLS="abl_dlv3p_cnx_s0 abl_dlv3p_cnx_s1 abl_dlv3p_cnx_s2 abl_segformer_b2_s0 abl_segformer_b2_s1 abl_segformer_b2_s2"
$PY scripts/benchmark.py $EXT --ablation --jitter --models $PRIMARY $CONTROLS --out results/clawtest/abl
$PY scripts/downscale_and_detector.py detector $EXT --models $CMP unet_cnx nnunet_nomirror $NNC $PRIMARY $CONTROLS --out clawtest/detector_per_hoof.csv
$PY scripts/benchmark.py $EXT --models yolo_roi_c010 yolo_roi_c005 yolo_roi_c001 rfdetr_roi_c010 rfdetr_roi_c005 rfdetr_roi_c001 --out results/clawtest/lowconf

# data-matched baselines on both test sets
MM="yolo_roi_matched yolo_roi_matched_c001 sam_ftm_box sam_ftm_halves sam_ftm_claws"
$PY scripts/benchmark.py --models $MM --out results/matched --repeats 1
$PY scripts/benchmark.py --models yolo_roi_matched yolo_roi_matched_c001 sam_ftm_halves --jitter --out results/matched
$PY scripts/benchmark.py $EXT --models $MM --out results/clawtest/matched --repeats 1
$PY scripts/benchmark.py $EXT --models yolo_roi_matched yolo_roi_matched_c001 sam_ftm_halves --jitter --out results/clawtest/matched

# 9. statistics
$PY scripts/stats_internal.py
$PY scripts/stats_variants.py
$PY scripts/stats_external.py
