# VITS — Vision-Informed Time Series Anomaly Detection

Companion repository for a benchmark-scale audit of frozen-vision time-series anomaly detection.
It contains the code, configurations and per-series result artifacts behind every number in the
paper. The paper is under review; publication details will be added here afterwards.

## Reproducing the audit without a GPU or the datasets

Every number in the paper is backed by per-series JSON under `artifacts/`. The scripts below read
only `artifacts/`, so they run on a fresh clone with the Python environment installed
(`pip install -e .[full]`) and need neither a GPU nor the raw datasets.

```bash
python scripts/make_figures.py --out figures   # Figures: baseline inversion, amplitude law, cost scaling
python scripts/summarize_ucr.py                     # UCR archive numbers -> artifacts/ucr/paper_summary.json
python scripts/holm_paired_family.py                # Holm correction -> artifacts/stats/holm_paired_family.json
pytest tests/ -q -m "not slow"                       # 355 tests
```

| Paper element | Artifact | Produced by (GPU / data needed) |
|---|---|---|
| Classic benchmarks (raw control on SMD) | `artifacts/raw_mahalanobis/` | `scripts/run_raw_mahalanobis_baseline.py` |
| Five-seed headline run | `artifacts/multiseed/headline_multiseed.json` | `scripts/run_headline_multiseed.py` |
| Raw controls on 1,044 TSB-AD series | `artifacts/tsb_ad_raw_audit/` | `scripts/run_tsb_ad_raw_audit.py` (CPU) |
| Rendered vs raw on the paired subset | `artifacts/tsb_ad_vision_paired/` | `scripts/run_tsb_ad_vision_paired.py` |
| ViT4TS head-to-head | `artifacts/vit4ts_paired/` | `scripts/run_vit4ts_paired.py` |
| Trained detectors (USAD, TimesNet, Anomaly Transformer) | `artifacts/tsb_ad_trained/` | `scripts/run_tsb_ad_trained_baselines.py` |
| PatchTrAD | `artifacts/tsb_ad_patchtrad/` | `scripts/run_tsb_ad_patchtrad.py` (needs `PATCHTRAD_DIR`) |
| TSB-AD pool detectors with TSB-AD's tuned hyperparameters | `artifacts/tsb_ad_pool/` | `scripts/run_tsb_ad_pool.py --detector <name>` |
| Random-init encoder control | `artifacts/random_backbone/` | `scripts/run_random_backbone_control.py` |
| Stride 1 vs 10 | `artifacts/stride_sanity/` | `scripts/run_stride_sanity.py` (CPU) |
| Cost scaling and crossover | `artifacts/cost_scaling/` | `scripts/measure_cost_scaling.py` |
| UCR archive evaluation | `artifacts/ucr/` | `scripts/summarize_ucr.py` (from the released per-series results) |
| Synthetic factorial | `artifacts/synthetic/` | `scripts/controlled_regime_full.py` |
| Label-free proxy search, frozen rule | `artifacts/regime_proxy/` | `scripts/regime_proxy_search.py` |
| Held-out confirmatory run | `artifacts/regime_confirmatory/` | `scripts/run_confirmatory_regime.py` |
| Multiple-comparison correction | `artifacts/stats/` | `scripts/holm_paired_family.py` |

Numbers produced on the classic benchmarks by the trained deep baselines (CATCH, GPT4TS, TimesNet,
Anomaly Transformer, USAD) require the raw datasets and a GPU; their summary values are reported in
the paper but their per-series outputs are not shipped.

### Datasets that need manual access

SWaT and WADI are distributed by iTrust under an access agreement and are not downloadable by
script. `src/data/swat.py` expects the official SWaT files placed under `data/raw/swat/`; the
remaining datasets are fetched by `scripts/download_{smd,psm,msl_smap}.py`, and TSB-AD from its
official release.

## The rendered detector

VITS renders sliding time-series windows as images, extracts patch tokens from
a frozen DINOv2 backbone, and scores anomalies via dual-signal fusion of
Mahalanobis distributional distance (primary) and trajectory prediction
residuals (regularizer).

## Installation

```bash
conda create -n vits python=3.10 -y && conda activate vits
pip install -e .[full]
```

The `full` extra installs the vision backbone (transformers, timm, Pillow) and
the dev tools (pytest, ruff, mypy). Use `pip install -e .[vision]` for
inference-only or `pip install -e .[dev]` for development without the backbone.

## Quick start

```bash
# 1. Train PatchTraj on the default dataset (SMD entity machine-1-1, line plot)
python scripts/train_patchtraj.py

# 2. Run detection with the trained checkpoint
python scripts/detect.py

# 3. Override any Hydra config from the CLI
python scripts/detect.py data=psm render=line_plot \
    scoring.dual_signal.alpha=0.1 scoring.dual_signal.auto_alpha=false
```

After installation no `PYTHONPATH` is required — `pip install -e .` registers
`src/` as the `vits` package.

## Legacy: method-paper tables (needs the full results/ tree)

All paper tables are regenerated from the canonical artifact tree:

```bash
python scripts/regenerate_paper_tables.py     # Tables 1, 3, 4
python scripts/build_ucr_canonical.py         # 109 eligible UCR series
bash scripts/run_spatial_benchmark.sh         # 28-entity SMD benchmark (4 GPUs)
```

The `results/` directory layout is:

```
results/
├── benchmark_smd_spatial/      # 28 entities × {LP, RP} × seed_42
├── multiseed/                  # 5 seeds × {PSM, MSL, SMAP}
├── ucr_canonical/              # 109 UCR series (eligible_list.json + per_series/)
├── ablation/                   # alpha sweep, spatial-attention ablation
└── reports/                    # aggregated paper tables
```

## Tests

```bash
pytest tests/                       # full test suite
pytest tests/ -m "not slow"         # skip slow tests
pytest tests/ --cov=src             # coverage report
```

## Configuration

All hyperparameters use Hydra YAML:

- `configs/data/{smd,psm,msl,smap,ucr}.yaml` — dataset settings
- `configs/render/{line_plot,recurrence_plot,multi_view}.yaml` — renderer
- `configs/model/{dinov2_base,clip_base}.yaml` — vision backbone
- `configs/experiment/{patchtraj_default,patchtraj_spatial}.yaml` — full configs

## Citation

The paper is under review; a citation entry will be added once it is published.

## License

MIT — see `LICENSE` for details.
