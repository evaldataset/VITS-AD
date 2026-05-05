# VITS-AD

**A Regime-Aware Evaluation Suite for Frozen-Vision Time-Series Anomaly Detection**

This repository accompanies the NeurIPS 2026 Evaluations & Datasets (E&D)
Track submission. It contains the method source tree, Hydra configurations,
reproduction scripts, and the saved JSON ledgers underlying every claim in
the main paper and supplement.

The submission is **double-blind**; this repository is anonymous and routed
through `https://anonymous.4open.science` for reviewer access. Author and
institution information will be added upon acceptance.

## Contribution scope (E&D Track)

This is a **benchmark analysis and evaluation methodology** contribution. It
does not introduce a new dataset; it (i) introduces and audits a previously
missing baseline (raw-space Mahalanobis with Ledoit–Wolf shrinkage) for the
frozen-vision time-series anomaly detection (TSAD) family, (ii) delivers a
regime-aware evaluation protocol with paired Wilcoxon tests, multi-seed
bootstrap CIs, paired-99 UCR comparison, and dataset-level compute
accounting, and (iii) reports negative results on amplitude-dominated
multivariate regimes (SMD/PSM/MSL).

## Pipeline

```
Time-series window (W × D)
  → Renderer (line plot or recurrence plot, deterministic)
  → Frozen DINOv2-B/14 ViT  →  patch tokens (256 × 768)
  → Dual-signal scorer
       ├─ Mahalanobis distributional distance (primary, Ledoit–Wolf)
       └─ patch-trajectory residual (regularizer)
  → CalibGuard v3 (empirical FAR diagnostic, leak-free)
  → Anomaly score
```

## Installation

```bash
conda create -n vits-ad python=3.10 -y && conda activate vits-ad
pip install -e .[full]
```

Extras: `[vision]` (inference only), `[dev]` (tests + lint), `[full]` (both).
`pip install -e .` registers `src/` as the importable `vits` package.

## Quick start

```bash
# Train default (SMD entity machine-1-1, line plot, temporal-only)
python scripts/train_patchtraj.py

# Detect with the saved checkpoint
python scripts/detect.py

# Renderer-adaptive: spatial+dual on LP, temporal-only on RP
python scripts/train_patchtraj.py data=smd render=line_plot \
    patchtraj.spatial_attention=true \
    scoring.dual_signal.enabled=true scoring.dual_signal.alpha=0.1
python scripts/detect.py data=smd render=line_plot \
    patchtraj.spatial_attention=true \
    scoring.dual_signal.enabled=true scoring.dual_signal.alpha=0.1
```

## Reproducing paper claims

```bash
# 1. Raw-space Mahalanobis baseline (Tables 1–2 raw rows)
python scripts/run_raw_mahalanobis_baseline.py

# 2. Multi-seed runs (PSM/MSL/SMAP, 5 seeds)
python scripts/run_multiseed.py

# 3. Paired-99 UCR comparison
python scripts/build_ucr_canonical.py
python scripts/run_ucr_experiment.py

# 4. Statistical tests (paired Wilcoxon, Cohen's d)
python scripts/statistical_tests.py
python scripts/statistical_tests_multiseed.py

# 5. CalibGuard v3 (leak-free, empirical FAR diagnostic)
python scripts/run_calibguard_v3.py

# 6. Compute disclosure
python scripts/run_fps_benchmark.py

# 7. Regenerate paper tables and figures
python scripts/regenerate_paper_tables.py
python scripts/generate_paper_figures.py
```

A top-level driver that chains the above is `scripts/reproduce_paper.sh`.

## Datasets (existing public benchmarks; not redistributed)

| Benchmark | Source                                              |
|-----------|------------------------------------------------------|
| SMD       | NetManAIOps / OmniAnomaly release                    |
| PSM       | eBay RANSynCoders sample data                        |
| SMAP, MSL | NASA JPL via `khundman/telemanom`                    |
| UCR       | UCR Anomaly Archive (109 series, paired 99 reported) |

Download helpers: `scripts/download_smd.py`, `scripts/download_psm.py`,
`scripts/download_msl_smap.py`. The UCR archive must be obtained from its
official source.

## Tests

```bash
pytest tests/                       # full suite
pytest tests/ -m "not slow"         # skip slow tests
pytest tests/ --cov=src             # coverage
ruff check src/ tests/ scripts/     # lint
mypy src/                            # type check
```

## Repository layout

```
src/         method package (renderers, backbone, predictors, scorers, calibration)
configs/     Hydra YAML (data / model / render / experiment groups)
scripts/     training, detection, baselines, statistical tests, reproduction
tests/       pytest suite (deterministic rendering, leak-free calibration, scorer math)
```

## Configuration

All hyperparameters are Hydra YAML; override anything from the CLI:

```bash
python scripts/detect.py data=psm render=recurrence_plot \
    model.pretrained=facebook/dinov2-base \
    scoring.dual_signal.alpha=0.2 \
    scoring.smooth_window=21
```

Key config groups:

- `configs/data/{smd,psm,msl,smap,ucr}.yaml`
- `configs/render/{line_plot,recurrence_plot,gaf,multi_view}.yaml`
- `configs/model/{dinov2_base,clip_base,siglip_base}.yaml`
- `configs/experiment/{vits_ad_default,vits_ad_dual_only,vits_ad_improved,vits_ad_spatial,vits_ad_sweep}.yaml`

## Citation

```bibtex
@inproceedings{vitsad2026,
  title  = {{VITS-AD}: A Regime-Aware Evaluation Suite for Frozen-Vision Time-Series Anomaly Detection},
  author = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year   = {2026}
}
```

## License

MIT — see `LICENSE`.
