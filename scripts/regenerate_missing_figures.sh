#!/usr/bin/env bash
# Regenerate the three manuscript figures that were previously pulled from the
# dead .sisyphus / results NAS symlinks. Run AFTER the data/ and results/ NAS
# mounts are restored. Outputs land in paper/figures/ with the exact filenames
# the manuscript expects:
#   figures/fig2_entity_heatmap.pdf
#   figures/fig3_ablation.pdf
#   figures/temporal_saliency_overlay.pdf
#
# Prereqs (must exist once NAS is mounted):
#   results/reports/ablation_results.csv        (fig3_ablation)
#   results/benchmark_smd_spatial/...           (fig2_entity_heatmap)
#   a trained PatchTraj checkpoint + a dataset  (temporal_saliency_overlay)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] fig2_entity_heatmap + fig3_ablation (+ other paper figures)"
python3 scripts/generate_paper_figures.py   # now writes into paper/figures/

echo "[2/3] temporal_saliency_overlay"
# Adjust --output-dir / checkpoint / data to your environment; the script
# auto-resolves the checkpoint under the given root. Overlay is saved as
# temporal_saliency_overlay.{pdf,png} in the output dir.
python3 scripts/run_temporal_saliency.py --output-dir paper/figures data=smap render=line_plot

echo "[3/3] verifying the three expected figures exist"
missing=0
for f in paper/figures/fig2_entity_heatmap.pdf \
         paper/figures/fig3_ablation.pdf \
         paper/figures/temporal_saliency_overlay.pdf; do
  if [ -f "$f" ]; then echo "  OK   $f"; else echo "  MISSING $f"; missing=1; fi
done
[ "$missing" -eq 0 ] && echo "All three figures present. Now: cd paper && latexmk -pdf neurips_submission.tex"
exit "$missing"
