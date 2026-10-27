#!/usr/bin/env bash
set -euo pipefail

# Reproduce the published ROSE-SAM3 main-table protocol.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${SAM3_CHECKPOINT:?Set SAM3_CHECKPOINT to the frozen SAM 3 checkpoint}"
: "${SAM3_BPE_PATH:?Set SAM3_BPE_PATH to the SAM 3 BPE vocabulary}"

OUT_ROOT="${MAIN_RUN_ROOT:-$ROOT_DIR/work_dirs/main_table}"

declare -A CONFIGS=(
  [openearthmap]="./configs/cfg_openearthmap.py"
  [udd5]="./configs/cfg_udd5.py"
  [vdd]="./configs/cfg_vdd.py"
  [uavid]="./configs/cfg_uavid.py"
  [vaihingen]="./configs/cfg_vaihingen.py"
  [potsdam]="./configs/cfg_potsdam.py"
  [loveda]="./configs/cfg_loveda.py"
  [isaid]="./configs/cfg_iSAID.py"
)


DATASETS=(openearthmap udd5 vdd uavid vaihingen potsdam loveda isaid)
if [[ "$#" -gt 0 ]]; then DATASETS=("$@"); fi

for dataset in "${DATASETS[@]}"; do
  cfg="${CONFIGS[$dataset]:-}"
  [[ -n "$cfg" ]] || { echo "Unsupported dataset: $dataset" >&2; exit 2; }
  prompt_file="$ROOT_DIR/calibrated_prompts/final_promptbank/cls_${dataset}.txt"
  prompt_source="frozen_release_bank"
  dag_work_dir="${DAG_WORK_ROOT:-$OUT_ROOT/promptbank_rebuild}/$dataset"
  if [[ ! -s "$prompt_file" ]]; then
    dataset_upper="${dataset^^}"
    calibration_var="CALIBRATION_DIR_${dataset_upper}"
    calibration_dir="${!calibration_var:-}"
    if [[ -z "$calibration_dir" && -n "${CALIBRATION_ROOT:-}" ]]; then
      calibration_dir="$CALIBRATION_ROOT/$dataset"
    fi
    if [[ -z "$calibration_dir" ]]; then
      root_var="${dataset_upper}_ROOT"
      dataset_root="${!root_var:-}"
      if [[ -n "$dataset_root" ]]; then
        case "$dataset" in
          udd5|vdd) calibration_dir="$dataset_root/train/src" ;;
          *) calibration_dir="$dataset_root/img_dir/train" ;;
        esac
      fi
    fi
    if [[ -z "$calibration_dir" ]]; then
      echo "A DAG run is required to rebuild the missing PromptBank." >&2
      echo "Set ${calibration_var} to this dataset's unlabeled calibration-image directory," >&2
      echo "or set CALIBRATION_ROOT to a directory containing a '$dataset' subdirectory." >&2
      exit 2
    fi
    dag_args=(
      --dataset "$dataset"
      --calibration-image-dir "$calibration_dir"
      --work-dir "$dag_work_dir"
    )
    echo "========== $dataset: running response-guided DAG PromptBank search =========="
    python scripts/rebuild_promptbank.py "${dag_args[@]}"
    prompt_source="dag_search"
  fi
  # Verification is read-only. A missing bank is never silently replaced by a
  # copied artifact; the response-guided DAG above must produce it first.
  python scripts/verify_promptbank.py --dataset "$dataset"
  [[ -s "$prompt_file" ]] || { echo "Frozen PromptBank unavailable: $prompt_file" >&2; exit 1; }

  run_dir="$OUT_ROOT/$dataset"
  mkdir -p "$run_dir"
  cat > "$run_dir/protocol.env" <<EOF
release_id=ROSE-SAM3
dataset=$dataset
prompt_file=$prompt_file
prompt_source=$prompt_source
inference=SICH
EOF

  echo "========== $dataset =========="
  ROSE_WORK_DIR="$run_dir" \
    python eval_rose_sam3.py "$cfg" \
      2>&1 | tee "$run_dir/eval.log"
done

echo "Main-table evaluation outputs: $OUT_ROOT"
