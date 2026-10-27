#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SAM3_CHECKPOINT:?Set SAM3_CHECKPOINT to the frozen SAM 3 checkpoint}"
: "${SAM3_BPE_PATH:?Set SAM3_BPE_PATH to the SAM 3 BPE vocabulary}"

CORRUPTED_ROOT="${CORRUPTED_ROOT:-$ROOT/work_dirs/rs_ovss_c/images}"
OUT_ROOT="${CORRUPTION_RUN_ROOT:-$ROOT/work_dirs/rs_ovss_c/evaluation}"
CORRUPTIONS="${CORRUPTIONS:-gaussian motion_blur downsample fog shadow stripe}"
SEVERITY="${SEVERITY:-3}"
DATASETS=(openearthmap loveda isaid potsdam vaihingen uavid udd5 vdd)

if [[ "$#" -gt 0 ]]; then DATASETS=("$@"); fi

config_for() {
  case "$1" in
    openearthmap) echo "./configs/cfg_openearthmap.py" ;;
    loveda)       echo "./configs/cfg_loveda.py" ;;
    isaid)        echo "./configs/cfg_iSAID.py" ;;
    potsdam)      echo "./configs/cfg_potsdam.py" ;;
    vaihingen)    echo "./configs/cfg_vaihingen.py" ;;
    uavid)        echo "./configs/cfg_uavid.py" ;;
    udd5)         echo "./configs/cfg_udd5.py" ;;
    vdd)          echo "./configs/cfg_vdd.py" ;;
    *) echo "Unsupported dataset: $1" >&2; return 2 ;;
  esac
}


for corruption in $CORRUPTIONS; do
  condition="${corruption}_s${SEVERITY}"
  for dataset in "${DATASETS[@]}"; do
    dataset_upper="${dataset^^}"
    root_var="${dataset_upper}_ROOT"
    dataset_root="${!root_var:-}"
    [[ -n "$dataset_root" ]] || { echo "Set ${root_var} to the dataset root." >&2; exit 2; }
    image_dir="$CORRUPTED_ROOT/$dataset/$condition"
    [[ -d "$image_dir" ]] || { echo "Missing corrupted image directory: $image_dir" >&2; exit 2; }
    prompt_file="$ROOT/calibrated_prompts/final_promptbank/cls_${dataset}.txt"
    python scripts/verify_promptbank.py --dataset "$dataset"
    run_dir="$OUT_ROOT/$dataset/$condition"
    mkdir -p "$run_dir"

    echo "[EVAL] dataset=$dataset condition=$condition"
    ROSE_WORK_DIR="$run_dir" python eval_rose_sam3.py "$(config_for "$dataset")" \
      --cfg-options \
      test_dataloader.dataset.data_prefix.img_path="$image_dir" \
      test_dataloader.dataset.img_suffix=.png \
      2>&1 | tee "$run_dir/eval.log"
  done
done

echo "Corruption evaluation outputs: $OUT_ROOT"
