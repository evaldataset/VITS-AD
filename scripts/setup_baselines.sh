#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="${ROOT_DIR}/baselines"
TIMESNET_DIR="${BASELINES_DIR}/TimesNet"
CATCH_DIR="${BASELINES_DIR}/CATCH"

log() {
  printf "%s [INFO] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf "%s [WARN] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

clone_if_missing() {
  local repo_url="$1"
  local dest_dir="$2"

  if [[ -d "${dest_dir}/.git" ]]; then
    log "Repository already exists at ${dest_dir}; skipping clone."
    return
  fi

  if [[ -d "${dest_dir}" ]] && [[ -n "$(ls -A "${dest_dir}")" ]]; then
    warn "Directory ${dest_dir} exists but is not a git clone; skipping clone."
    return
  fi

  mkdir -p "$(dirname "${dest_dir}")"
  log "Cloning ${repo_url} into ${dest_dir}"
  git clone "${repo_url}" "${dest_dir}"
}

install_requirements_if_present() {
  local repo_dir="$1"

  if [[ ! -f "${repo_dir}/requirements.txt" ]]; then
    warn "No requirements.txt found in ${repo_dir}; skipping pip install."
    return
  fi

  log "Installing requirements for ${repo_dir}"
  python -m pip install -r "${repo_dir}/requirements.txt" || warn "pip install failed for ${repo_dir}; continuing with existing environment."
}

ensure_timesnet_datasets() {
  local dataset_dir="${TIMESNET_DIR}/dataset"
  mkdir -p "${dataset_dir}/SMD" "${dataset_dir}/PSM" "${dataset_dir}/MSL" "${dataset_dir}/SMAP"

  local -a required_files=(
    "${dataset_dir}/SMD/SMD_train.npy"
    "${dataset_dir}/SMD/SMD_test.npy"
    "${dataset_dir}/SMD/SMD_test_label.npy"
    "${dataset_dir}/PSM/train.csv"
    "${dataset_dir}/PSM/test.csv"
    "${dataset_dir}/PSM/test_label.csv"
    "${dataset_dir}/MSL/MSL_train.npy"
    "${dataset_dir}/MSL/MSL_test.npy"
    "${dataset_dir}/MSL/MSL_test_label.npy"
    "${dataset_dir}/SMAP/SMAP_train.npy"
    "${dataset_dir}/SMAP/SMAP_test.npy"
    "${dataset_dir}/SMAP/SMAP_test_label.npy"
  )

  local missing_count=0
  local file_path
  for file_path in "${required_files[@]}"; do
    if [[ ! -f "${file_path}" ]]; then
      missing_count=$((missing_count + 1))
    fi
  done

  if [[ "${missing_count}" -eq 0 ]]; then
    log "TimesNet datasets already available under ${dataset_dir}; skipping download."
    return
  fi

  log "Downloading missing TimesNet datasets from Hugging Face (${missing_count} files missing)."
  VITS_ROOT="${ROOT_DIR}" python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = "thuml/Time-Series-Library"
repo_type = "dataset"

dataset_files = {
    "SMD": ["SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy"],
    "PSM": ["train.csv", "test.csv", "test_label.csv"],
    "MSL": ["MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"],
    "SMAP": ["SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"],
}

root = Path(os.environ["VITS_ROOT"]) / "baselines" / "TimesNet" / "dataset"
root.mkdir(parents=True, exist_ok=True)

for dataset_name, file_names in dataset_files.items():
    target_dir = root / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        target_path = target_dir / file_name
        if target_path.exists():
            continue
        source_name = f"{dataset_name}/{file_name}"
        local_path = Path(
            hf_hub_download(repo_id=repo_id, filename=source_name, repo_type=repo_type)
        )
        target_path.write_bytes(local_path.read_bytes())
        print(f"[INFO] Downloaded {source_name} -> {target_path}")
PY
}

main() {
  mkdir -p "${BASELINES_DIR}"

  clone_if_missing "https://github.com/thuml/Time-Series-Library.git" "${TIMESNET_DIR}"
  clone_if_missing "https://github.com/decisionintelligence/CATCH.git" "${CATCH_DIR}"

  install_requirements_if_present "${TIMESNET_DIR}"
  install_requirements_if_present "${CATCH_DIR}"

  ensure_timesnet_datasets
  log "Baseline setup complete."
}

main "$@"
