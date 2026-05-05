#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${ROOT_DIR}/results/reports"
RUN_DIR="${ROOT_DIR}/results/reports/dl_baselines_${TIMESTAMP}"
SUMMARY_CSV="${REPORT_DIR}/dl_baselines_summary.csv"
TIMESTAMPED_CSV="${REPORT_DIR}/dl_baselines_${TIMESTAMP}.csv"
FAIL_LOG="${RUN_DIR}/failed_runs.log"

log() {
  printf "%s [INFO] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf "%s [WARN] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

append_result_row() {
  local dataset="$1"
  local method="$2"
  local json_path="$3"

  python - "$dataset" "$method" "$json_path" "$SUMMARY_CSV" "$TIMESTAMPED_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
method = sys.argv[2]
json_path = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])
timestamped_csv = Path(sys.argv[5])

payload = json.loads(json_path.read_text(encoding="utf-8"))
metrics = payload["metrics"]
row = {
    "dataset": dataset,
    "method": method,
    "auc_roc": float(metrics["auc_roc"]),
    "auc_pr": float(metrics["auc_pr"]),
    "f1_pa": float(metrics["f1_pa"]),
}

for path in (summary_csv, timestamped_csv):
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "auc_roc", "auc_pr", "f1_pa"],
        )
        writer.writerow(row)
PY
}

mkdir -p "$REPORT_DIR" "$RUN_DIR"

printf "dataset,method,auc_roc,auc_pr,f1_pa\n" > "$SUMMARY_CSV"
printf "dataset,method,auc_roc,auc_pr,f1_pa\n" > "$TIMESTAMPED_CSV"
: > "$FAIL_LOG"

DATASETS=(smd psm msl smap)
METHODS=(timesnet catch)

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

for method in "${METHODS[@]}"; do
  case "$method" in
    timesnet)
      runner="${ROOT_DIR}/scripts/run_timesnet_baseline.py"
      ;;
    catch)
      runner="${ROOT_DIR}/scripts/run_catch_baseline.py"
      ;;
    *)
      warn "Unknown method ${method}; skipping."
      continue
      ;;
  esac

  if [[ ! -f "$runner" ]]; then
    warn "Runner not found for method=${method}: ${runner}"
    printf "%s,%s,%s\n" "all" "$method" "missing runner: ${runner}" >> "$FAIL_LOG"
    continue
  fi

  for dataset in "${DATASETS[@]}"; do
    run_output_dir="${RUN_DIR}/${method}/${dataset}"
    mkdir -p "$run_output_dir"

    log "Running method=${method} dataset=${dataset} gpu=${GPU_ID}"
    if python "$runner" --dataset "$dataset" --gpu "$GPU_ID" --output_dir "$run_output_dir"; then
      result_json="${run_output_dir}/${method}_results.json"
      if [[ ! -f "$result_json" ]]; then
        warn "Result JSON missing after successful run: $result_json"
        printf "%s,%s,%s\n" "$dataset" "$method" "missing result json" >> "$FAIL_LOG"
        continue
      fi
      append_result_row "$dataset" "$method" "$result_json"
      log "Completed method=${method} dataset=${dataset}"
    else
      warn "Failed method=${method} dataset=${dataset}; continuing."
      printf "%s,%s,%s\n" "$dataset" "$method" "run failed" >> "$FAIL_LOG"
    fi
  done
done

log "Wrote summary CSV: ${SUMMARY_CSV}"
log "Wrote timestamped CSV: ${TIMESTAMPED_CSV}"
if [[ -s "$FAIL_LOG" ]]; then
  warn "Some runs failed. See ${FAIL_LOG}"
else
  log "All configured runs completed without failures."
fi
