#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CORRUPTED_ROOT="${CORRUPTED_ROOT:-$ROOT/work_dirs/rs_ovss_c/images}"
IMAGE_SPLIT="${IMAGE_SPLIT:-val}"
CORRUPTIONS="${CORRUPTIONS:-gaussian motion_blur downsample fog shadow stripe}"
SEVERITY="${SEVERITY:-3}"
CORRUPTION_SEED="${CORRUPTION_SEED:-3407}"
DATASETS=(openearthmap loveda isaid potsdam vaihingen uavid udd5 vdd)

if [[ "$#" -gt 0 ]]; then DATASETS=("$@"); fi

for dataset in "${DATASETS[@]}"; do
  dataset_upper="${dataset^^}"
  root_var="${dataset_upper}_ROOT"
  dataset_root="${!root_var:-}"
  if [[ -z "$dataset_root" ]]; then
    echo "Set ${root_var} to the dataset root before generating corruptions." >&2
    exit 2
  fi
  # Keep the source split aligned with the released evaluation configs.
  # IMAGE_SPLIT applies to datasets whose configs use img_dir/<split>.
  case "$dataset" in
    openearthmap|loveda|isaid|potsdam|vaihingen)
      source_dir="$dataset_root/img_dir/$IMAGE_SPLIT"
      ;;
    uavid)
      source_dir="$dataset_root/img_dir/test"
      ;;
    udd5)
      source_dir="$dataset_root/val/src"
      ;;
    vdd)
      source_dir="$dataset_root/test/src"
      ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      exit 2
      ;;
  esac
  if [[ ! -d "$source_dir" ]]; then
    echo "Image split does not exist: $source_dir" >&2
    exit 2
  fi
  echo "[BUILD] dataset=$dataset source=$source_dir severity=$SEVERITY seed=$CORRUPTION_SEED"
  python -m rs_ovss_c.generate "$source_dir" "$CORRUPTED_ROOT/$dataset" \
    --corruptions "$CORRUPTIONS" \
    --severity "$SEVERITY" \
    --seed "$CORRUPTION_SEED"
done

echo "Corrupted RGB images are under: $CORRUPTED_ROOT"
echo "Original annotation directories were not read or modified."
