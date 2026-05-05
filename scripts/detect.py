"""PatchTraj inference and anomaly detection script."""

from __future__ import annotations

# pyright: reportMissingImports=false,reportMissingTypeArgument=false,reportUnknownVariableType=false,reportUnknownMemberType=false,reportUnknownArgumentType=false,reportUnknownParameterType=false,reportImplicitStringConcatenation=false,reportImplicitOverride=false

import json
import logging
import sys
from pathlib import Path
from typing import Any, Tuple, TypeAlias

import hydra
import numpy as np
import numpy.typing as npt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.base import create_sliding_windows, normalize_data, time_based_split
from src.data.msl import _load_msl_labels, _load_msl_matrix
from src.data.psm import _load_psm_features, _load_psm_labels
from src.data.smap import _load_smap_labels, _load_smap_matrix
from src.data.smd import _load_smd_labels, _load_smd_matrix
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.models.patchtraj import PatchTrajPredictor, SpatialTemporalPatchTrajPredictor
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.gaf import render_gaf_batch
from src.rendering.recurrence_plot import render_recurrence_plot_batch
from src.rendering.token_correspondence import (
    compute_correspondence_map,
    compute_identity_map,
)
from src.rendering.token_correspondence_ot import compute_ot_correspondence
from src.scoring.patchtraj_scorer import (
    compute_patchtraj_score,
    normalize_scores,
    smooth_scores,
)
from src.scoring.dual_signal_scorer import DualSignalScorer
from src.scoring.multiscale_ensemble import MultiScaleEnsemble
from src.utils.reproducibility import get_device, seed_everything
from src.scoring.calibguard import CalibGuard, compute_far_at_alpha

LOGGER = logging.getLogger(__name__)
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


class TokenSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Sequence dataset for PatchTraj inference."""

    def __init__(
        self,
        token_windows: FloatArray,
        window_labels: IntArray,
        k: int,
        delta: int,
    ) -> None:
        super().__init__()
        if token_windows.ndim != 3:
            raise ValueError(
                f"token_windows must have shape (T, N, D), got {token_windows.shape}."
            )
        if window_labels.ndim != 1:
            raise ValueError(
                f"window_labels must have shape (T,), got {window_labels.shape}."
            )
        if token_windows.shape[0] != window_labels.shape[0]:
            raise ValueError(
                "token_windows and window_labels must have the same length, got "
                f"{token_windows.shape[0]} and {window_labels.shape[0]}."
            )
        if k <= 0 or delta <= 0:
            raise ValueError(f"k and delta must be positive, got k={k}, delta={delta}.")
        if token_windows.shape[0] < (k + delta):
            raise ValueError(
                "Not enough windows for sequence construction: "
                f"T={token_windows.shape[0]}, required at least {k + delta}."
            )

        self._tokens = token_windows.astype(np.float32, copy=False)
        self._labels = window_labels.astype(np.int64, copy=False)
        self._k = int(k)
        self._delta = int(delta)
        self._num_samples = self._tokens.shape[0] - self._k - self._delta + 1

    def __len__(self) -> int:
        """Return number of samples."""
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sample as (input_seq, target_tokens, target_label)."""
        start = idx
        end = start + self._k
        target_idx = end + self._delta - 1
        return (
            torch.from_numpy(self._tokens[start:end]),
            torch.from_numpy(self._tokens[target_idx]),
            torch.tensor(self._labels[target_idx], dtype=torch.int64),
        )


def _load_full_series(cfg: DictConfig) -> tuple[FloatArray, IntArray, int | None]:
    """Load full train+test series and timestep labels from config."""
    dataset_name = str(cfg.data.name).strip().lower()
    raw_dir = Path(to_absolute_path(str(cfg.data.raw_dir)))
    official_test_start: int | None = None

    if dataset_name == "smd":
        entity = str(cfg.data.entity)
        train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
        test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")
        official_test_start = int(train_data.shape[0])
    elif dataset_name == "psm":
        train_data = _load_psm_features(raw_dir / "train.csv")
        test_data = _load_psm_features(raw_dir / "test.csv")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_psm_labels(raw_dir / "test_label.csv")
        official_test_start = int(train_data.shape[0])
    elif dataset_name == "msl":
        train_data = _load_msl_matrix(raw_dir / "MSL_train.npy")
        test_data = _load_msl_matrix(raw_dir / "MSL_test.npy")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_msl_labels(raw_dir / "MSL_test_label.npy")
        official_test_start = int(train_data.shape[0])
    elif dataset_name == "smap":
        train_data = _load_smap_matrix(raw_dir / "SMAP_train.npy")
        test_data = _load_smap_matrix(raw_dir / "SMAP_test.npy")
        train_labels = np.zeros((train_data.shape[0],), dtype=np.int64)
        test_labels = _load_smap_labels(raw_dir / "SMAP_test_label.npy")
        official_test_start = int(train_data.shape[0])
    else:
        raise ValueError(
            "Unsupported dataset "
            f"'{cfg.data.name}'. Expected one of ['smd', 'psm', 'msl', 'smap']."
        )

    if bool(cfg.data.normalize):
        train_data, test_data = normalize_data(
            train_data=train_data,
            test_data=test_data,
            method=str(cfg.data.norm_method),
        )
    return (
        np.concatenate([train_data, test_data], axis=0),
        np.concatenate([train_labels, test_labels], axis=0),
        official_test_start,
    )


def _compute_window_labels(labels: IntArray, window_size: int, stride: int) -> IntArray:
    """Compute window labels where any anomalous timestep marks the window."""
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}.")
    starts = np.arange(0, labels.shape[0] - window_size + 1, stride, dtype=np.int64)
    return np.asarray(
        [int(np.any(labels[start : start + window_size] == 1)) for start in starts],
        dtype=np.int64,
    )


def _get_batch_renderer(name: str) -> Any:
    """Get batch renderer function by name."""
    if name == "line_plot":
        return render_line_plot_batch
    if name == "gaf":
        return render_gaf_batch
    if name == "recurrence_plot":
        return render_recurrence_plot_batch
    raise ValueError(f"Unsupported render.name '{name}'.")


def _render_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Build renderer keyword arguments from config."""
    name = str(cfg.render.name)
    if name == "line_plot":
        return {
            "image_size": int(cfg.render.image_size),
            "dpi": int(cfg.render.dpi),
            "colormap": str(cfg.render.colormap),
            "line_width": float(cfg.render.line_width),
            "background_color": str(cfg.render.background_color),
            "show_axes": bool(cfg.render.show_axes),
            "show_grid": bool(cfg.render.show_grid),
        }
    if name == "gaf":
        return {
            "image_size": int(cfg.render.image_size),
            "method": str(cfg.render.method),
        }
    if name == "recurrence_plot":
        kwargs: dict[str, Any] = {
            "image_size": int(cfg.render.image_size),
            "metric": str(cfg.render.metric),
        }
        threshold_val = cfg.render.get("threshold", "auto")
        if threshold_val != "auto":
            kwargs["threshold"] = float(threshold_val)
        return kwargs
    raise ValueError(f"Unsupported render.name '{name}'.")


def _test_cache_meta(cfg: DictConfig, windows: FloatArray) -> dict[str, Any]:
    return {
        "dataset": str(cfg.data.name),
        "entity": str(cfg.data.get("entity", cfg.data.name)),
        "model": str(cfg.model.pretrained),
        "render": str(cfg.render.name),
        "window_size": int(cfg.data.window_size),
        "stride": int(cfg.data.stride),
        "image_size": int(cfg.render.image_size),
        "normalize": bool(cfg.data.normalize),
        "norm_method": str(cfg.data.norm_method),
        "num_windows": int(windows.shape[0]),
        "render_kwargs": _render_kwargs(cfg),
    }


def _extract_test_tokens(
    windows: FloatArray,
    cfg: DictConfig,
    backbone: VisionBackbone,
    output_dir: Path,
) -> FloatArray:
    """Render test windows, extract patch tokens, and cache results."""
    cache_path = output_dir / "test_patch_tokens.npy"
    cache_meta_path = output_dir / "test_patch_tokens_meta.json"
    expected_meta = _test_cache_meta(cfg=cfg, windows=windows)
    if cache_path.exists():
        cached = np.load(cache_path)
        cached_meta: dict[str, Any] | None = None
        if cache_meta_path.exists():
            with cache_meta_path.open("r", encoding="utf-8") as handle:
                cached_meta = json.load(handle)
        if cached.shape[0] == windows.shape[0] and cached_meta == expected_meta:
            LOGGER.info("Loading cached test patch tokens from %s", cache_path)
            return cached
        LOGGER.warning("Ignoring stale token cache at %s", cache_path)

    renderer_name = str(cfg.render.name)
    batch_render_fn = _get_batch_renderer(renderer_name)
    render_kw = _render_kwargs(cfg)
    batch_size = int(cfg.training.batch_size)

    # Multi-layer feature extraction (optional)
    feature_layers_cfg = cfg.model.get("feature_layers", None)
    use_multilayer = feature_layers_cfg is not None and len(feature_layers_cfg) > 0
    if use_multilayer:
        feature_layers = tuple(int(idx) for idx in feature_layers_cfg)
        LOGGER.info("Using multi-layer features: layers=%s", feature_layers)

    chunks: list[FloatArray] = []
    with torch.no_grad():
        for start in range(0, windows.shape[0], batch_size):
            end = min(start + batch_size, windows.shape[0])
            images = batch_render_fn(windows[start:end], **render_kw)
            if use_multilayer:
                chunks.append(
                    backbone.extract_multilayer_tokens_from_numpy(
                        images, layers=feature_layers
                    ).astype(np.float64, copy=False)
                )
            else:
                chunks.append(
                    backbone.extract_patch_tokens_from_numpy(images).astype(
                        np.float64, copy=False
                    )
                )
    tokens = np.concatenate(chunks, axis=0)
    np.save(cache_path, tokens)
    with cache_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_meta, handle, indent=2, sort_keys=True)
    LOGGER.info("Saved test patch token cache to %s", cache_path)
    return tokens


def _resolve_checkpoint_path(cfg: DictConfig, output_dir: Path) -> Path:
    """Resolve optional checkpoint override from Hydra config."""
    checkpoint_value = cfg.get("checkpoint", None)
    if checkpoint_value is None or str(checkpoint_value).strip() == "":
        return output_dir / "best_model.pt"
    return Path(to_absolute_path(str(checkpoint_value)))


def _run_calibguard(
    scores: FloatArray,
    labels: IntArray,
    cfg: DictConfig,
    output_dir: Path,
) -> None:
    # WARNING: This function calibrates on test-set normal scores.
    # Results are EXPLORATORY ONLY and must NOT be used for paper claims.
    # For claim-bearing FAR guarantees, use CalibGuardV3.from_train_split().
    """Run CalibGuard FAR evaluation as post-processing.

    Uses normal test scores for calibration, then evaluates FAR control.
    Reports metrics for each configured alpha level.

    Args:
        scores: Normalized anomaly scores, shape (T,).
        labels: Binary labels, shape (T,). 0=normal, 1=anomaly.
        cfg: CalibGuard config section with alpha_values, calibration_ratio.
        output_dir: Directory to save CalibGuard results.
    """
    alpha_values = list(cfg.get("alpha_values", [0.01, 0.05, 0.10]))
    calib_ratio = float(cfg.get("calibration_ratio", 0.5))
    rolling_window = int(cfg.get("rolling_window", 0))  # Fixed mode by default

    LOGGER.info(
        "Running CalibGuard FAR evaluation: alphas=%s, calib_ratio=%.2f, rolling=%d",
        alpha_values,
        calib_ratio,
        rolling_window,
    )
    LOGGER.warning(
        "*** EXPERIMENTAL ONLY *** CalibGuard evaluation is experimental only: "
        "current implementation calibrates on held-out test normals and is NOT "
        "claim-bearing. Do NOT use these results for paper claims."
    )

    calibguard_results: dict[str, Any] = {}
    for alpha in alpha_values:
        alpha_key = f"alpha_{alpha:.2f}"
        try:
            far_result = compute_far_at_alpha(
                scores=scores,
                labels=labels,
                alpha=alpha,
                calibration_ratio=calib_ratio,
            )
            calibguard_results[alpha_key] = far_result
            LOGGER.info(
                "CalibGuard alpha=%.3f: FAR=%.4f (target=%.3f), "
                + "coverage=%.4f, threshold=%.4f",
                alpha,
                far_result["actual_far"],
                far_result["target_far"],
                far_result["coverage"],
                far_result["threshold"],
            )
        except ValueError as exc:
            LOGGER.warning("CalibGuard alpha=%.3f failed: %s", alpha, exc)
            calibguard_results[alpha_key] = {"error": str(exc)}

    # Also run full CalibGuard with rolling mode for online evaluation
    if rolling_window > 0:
        normal_mask = labels == 0
        if normal_mask.sum() > 0:
            normal_scores = scores[normal_mask]
            n_calib = max(1, int(normal_scores.size * calib_ratio))
            calib_scores = normal_scores[:n_calib]

            for alpha in alpha_values:
                alpha_key_rolling = f"alpha_{alpha:.2f}_rolling"
                guard = CalibGuard(
                    alpha=alpha,
                    rolling_window=rolling_window,
                    drift_sigma=float(cfg.get("drift_sigma", 3.0)),
                )
                guard.fit(calib_scores)
                flags, p_values, thresholds = guard.predict_batch(scores)
                stats = guard.get_stats()
                calibguard_results[alpha_key_rolling] = {
                    "empirical_far": stats.empirical_far,
                    "n_alarms": stats.n_alarms,
                    "n_predictions": stats.n_predictions,
                    "drift_detected": stats.drift_detected,
                    "final_threshold": float(thresholds[-1]),
                }
                LOGGER.info(
                    "CalibGuard rolling alpha=%.3f: FAR=%.4f, alarms=%d/%d, drift=%s",
                    alpha,
                    stats.empirical_far,
                    stats.n_alarms,
                    stats.n_predictions,
                    stats.drift_detected,
                )

    # Save results
    calibguard_results["experimental_only"] = True
    calibguard_path = output_dir / "experimental_calibguard_results.json"
    with calibguard_path.open("w", encoding="utf-8") as handle:
        json.dump(calibguard_results, handle, indent=2, sort_keys=True)
    LOGGER.info("CalibGuard results saved to %s", calibguard_path)


def _run_multiscale_ensemble(cfg: DictConfig, output_dir: Path) -> None:
    results_dir_value = str(cfg.multiscale.get("results_dir", "")).strip()
    results_dir = (
        Path(to_absolute_path(results_dir_value)) if results_dir_value else output_dir
    )
    entity_override = cfg.multiscale.get("entity", None)
    entity_value = str(entity_override).strip() if entity_override is not None else ""
    if len(entity_value) == 0:
        entity_value = str(cfg.data.get("entity", "")).strip()
    entity = entity_value if entity_value else None
    renderers_cfg = cfg.multiscale.get("renderers", [])
    renderers = [str(renderer).strip() for renderer in renderers_cfg if str(renderer).strip()]

    ensemble = MultiScaleEnsemble(
        window_sizes=tuple(int(window_size) for window_size in cfg.multiscale.window_sizes)
    )
    entries = ensemble.find_score_entries(
        results_dir=results_dir,
        entity=entity,
        renderers=renderers or None,
    )
    normalized_scores, eval_labels = ensemble.combine(
        entries=entries,
        method=str(cfg.multiscale.method),
    )

    smooth_window = int(cfg.scoring.get("smooth_window", 5)) if "scoring" in cfg else 5
    smooth_method = (
        str(cfg.scoring.get("smooth_method", "mean")) if "scoring" in cfg else "mean"
    )
    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        normalized_scores = smooth_scores(
            normalized_scores,
            window_size=smooth_window,
            method=smooth_method,
        )
        LOGGER.info(
            "Applied multi-scale score smoothing: method=%s, window_size=%d",
            smooth_method,
            smooth_window,
        )

    normalized_scores = normalize_scores(normalized_scores, method="minmax")
    np.save(output_dir / "multiscale_ensemble_scores.npy", normalized_scores)
    if eval_labels is not None:
        np.save(output_dir / "multiscale_ensemble_labels.npy", eval_labels)
        metrics = compute_all_metrics(normalized_scores, eval_labels)
        with (output_dir / "multiscale_ensemble_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
        LOGGER.info("Multi-scale ensemble completed. Output directory: %s", output_dir)
        for metric_name, metric_value in metrics.items():
            LOGGER.info("%s=%.6f", metric_name, metric_value)
        return

    LOGGER.info(
        "Multi-scale ensemble completed without labels. Output directory: %s", output_dir
    )


def _apply_multiscale_cli_flag(argv: list[str]) -> list[str]:
    normalized_args: list[str] = []
    for arg in argv:
        if arg == "--multiscale":
            normalized_args.append("multiscale.enabled=true")
            continue
        normalized_args.append(arg)
    return normalized_args


@hydra.main(
    config_path="../configs",
    config_name="experiment/patchtraj_default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run PatchTraj test-time inference and evaluation."""
    logging.basicConfig(level=logging.INFO)
    seed_everything(int(cfg.training.seed))

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(cfg.multiscale.get("enabled", False)):
        _run_multiscale_ensemble(cfg=cfg, output_dir=output_dir)
        return

    full_data, full_labels, official_test_start = _load_full_series(cfg)
    if official_test_start is not None:
        test_data = full_data[official_test_start:]
        test_labels = full_labels[official_test_start:]
    else:
        train_ratio_cfg = cfg.data.get("train_ratio", None)
        if train_ratio_cfg is None:
            raise ValueError(
                "data.train_ratio must be set for datasets without official split."
            )
        _, _, test_data, test_labels = time_based_split(
            data=full_data,
            labels=full_labels,
            train_ratio=float(train_ratio_cfg),
        )

    window_size = int(cfg.data.window_size)
    stride = int(cfg.data.stride)
    test_windows, _ = create_sliding_windows(
        data=test_data,
        labels=test_labels,
        window_size=window_size,
        stride=stride,
    )
    window_labels = _compute_window_labels(test_labels, window_size, stride)

    device = get_device()
    backbone = VisionBackbone(model_name=str(cfg.model.pretrained), device=device)
    token_windows = _extract_test_tokens(test_windows, cfg, backbone, output_dir)

    sequence_dataset = TokenSequenceDataset(
        token_windows=token_windows,
        window_labels=window_labels,
        k=int(cfg.patchtraj.K),
        delta=int(cfg.patchtraj.delta),
    )
    dataloader = DataLoader(
        sequence_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    use_identity_pi = bool(cfg.patchtraj.get("use_identity_pi", False))
    correspondence_mode = str(
        cfg.patchtraj.get("correspondence_mode", "geometric")
    ).strip().lower()
    if use_identity_pi:
        LOGGER.info("Using identity correspondence map (ablation mode)")
        pi, valid_mask = compute_identity_map(patch_grid=backbone.patch_grid)
    elif correspondence_mode == "ot":
        if int(token_windows.shape[0]) < 2:
            raise ValueError("OT correspondence requires at least two token windows.")
        soft_pi, sinkhorn_iterations = compute_ot_correspondence(
            tokens_t=torch.from_numpy(token_windows[0]).to(
                device=device, dtype=torch.float32
            ),
            tokens_t1=torch.from_numpy(token_windows[1]).to(
                device=device, dtype=torch.float32
            ),
            reg=float(cfg.patchtraj.get("ot_reg", 0.1)),
            max_iterations=int(cfg.patchtraj.get("ot_max_iterations", 100)),
            tolerance=float(cfg.patchtraj.get("ot_tolerance", 1e-3)),
            hard_assignment=False,
        )
        if sinkhorn_iterations > int(cfg.patchtraj.get("ot_max_iterations", 100)):
            raise RuntimeError(
                "Sinkhorn exceeded configured iteration budget: "
                f"{sinkhorn_iterations}."
            )
        pi = (
            soft_pi.argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        valid_mask = np.ones_like(pi, dtype=bool)
        LOGGER.info(
            "Using OT correspondence map (reg=%.4f, iterations=%d)",
            float(cfg.patchtraj.get("ot_reg", 0.1)),
            sinkhorn_iterations,
        )
    elif correspondence_mode == "geometric":
        pi, valid_mask = compute_correspondence_map(
            renderer_type=str(cfg.render.name),
            window_size=window_size,
            stride=stride,
            patch_grid=backbone.patch_grid,
        )
    else:
        raise ValueError(
            "patchtraj.correspondence_mode must be one of "
            f"['geometric', 'ot'], got '{correspondence_mode}'."
        )

    use_spatial = bool(cfg.patchtraj.get("spatial_attention", False))
    if use_spatial:
        model: nn.Module = SpatialTemporalPatchTrajPredictor(
            hidden_dim=int(backbone.hidden_dim),
            d_model=int(cfg.patchtraj.d_model),
            n_heads=int(cfg.patchtraj.n_heads),
            n_layers=int(cfg.patchtraj.n_layers),
            dim_feedforward=int(cfg.patchtraj.dim_feedforward),
            dropout=float(cfg.patchtraj.dropout),
            patch_grid=backbone.patch_grid,
        ).to(device)
    else:
        model = PatchTrajPredictor(
            hidden_dim=int(backbone.hidden_dim),
            d_model=int(cfg.patchtraj.d_model),
            n_heads=int(cfg.patchtraj.n_heads),
            n_layers=int(cfg.patchtraj.n_layers),
            dim_feedforward=int(cfg.patchtraj.dim_feedforward),
            dropout=float(cfg.patchtraj.dropout),
            activation=str(cfg.patchtraj.activation),
        ).to(device)

    checkpoint_path = _resolve_checkpoint_path(cfg, output_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint format invalid: expected state_dict mapping.")
    model.load_state_dict(state_dict)
    model.eval()

    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for token_seq, actual_tokens, target_labels in dataloader:
            token_seq = token_seq.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            actual_tokens = actual_tokens.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            predicted_tokens = model(token_seq)
            batch_scores = compute_patchtraj_score(
                predicted_tokens=predicted_tokens,
                actual_tokens=actual_tokens,
                pi=pi,
                valid_mask=valid_mask,
            )
            score_chunks.append(batch_scores.detach().cpu().numpy().astype(np.float64))
            label_chunks.append(target_labels.numpy().astype(np.int64, copy=False))

    raw_scores = np.concatenate(score_chunks, axis=0)
    eval_labels = np.concatenate(label_chunks, axis=0)

    # --- Dual-Signal Scoring (trajectory + distributional distance) ---
    dual_cfg = cfg.get("scoring", {}).get("dual_signal", {})
    if bool(dual_cfg.get("enabled", False)):
        dual_state = (
            checkpoint.get("dual_signal_state")
            if isinstance(checkpoint, dict)
            else None
        )
        if dual_state is not None:
            dual_alpha = float(dual_cfg.get("alpha", 0.5))
            dual_scorer = DualSignalScorer(alpha=dual_alpha)
            dual_scorer.load_state_dict(dual_state)

            # --- Validation-based alpha selection ---
            # Skip if alpha is explicitly overridden via config (for ablation)
            # Default to False so that running detect.py without an explicit
            # config does not silently invoke the CoV selector.  The selector
            # is documented in the paper only as a negative-result diagnostic
            # (see Section sec:smd-ablations); production runs use the
            # per-dataset fixed alpha specified in the experiment config.
            auto_alpha = bool(dual_cfg.get("auto_alpha", False))
            if not auto_alpha:
                dual_scorer.alpha = dual_alpha
                LOGGER.info(
                    "Alpha fixed from config: %.2f (auto_alpha=false).",
                    dual_alpha,
                )
            if auto_alpha:
              val_traj = (
                  checkpoint.get("val_traj_scores")
                  if isinstance(checkpoint, dict)
                  else None
              )
              val_dist = (
                  checkpoint.get("val_dist_scores")
                  if isinstance(checkpoint, dict)
                  else None
              )
            else:
              val_traj, val_dist = None, None
            if val_traj is not None and val_dist is not None:
                val_traj = np.asarray(val_traj, dtype=np.float64)
                val_dist = np.asarray(val_dist, dtype=np.float64)
                alpha_candidates = np.arange(0.0, 1.05, 0.1)
                best_alpha = dual_alpha
                best_neg_entropy = -np.inf
                for a_candidate in alpha_candidates:
                    a_candidate = float(round(a_candidate, 2))
                    trial_scorer = DualSignalScorer(alpha=a_candidate)
                    trial_scorer.load_state_dict(dual_state)
                    trial_scorer.alpha = a_candidate
                    fused_val = trial_scorer.fuse(
                        traj_scores=val_traj, dist_scores=val_dist,
                    )
                    # Validation data is all normal. Select alpha that maximises
                    # score spread (std / mean of absolute fused scores).  Higher
                    # spread means the scoring function is more sensitive to
                    # deviations and thus more likely to separate normal from
                    # anomalous on test data.  This is a proxy criterion that
                    # uses ONLY training-split validation data.
                    fused_abs = np.abs(fused_val)
                    mu = float(np.mean(fused_abs))
                    std = float(np.std(fused_abs))
                    # Coefficient of variation — higher = more discriminative
                    spread = std / max(mu, 1e-12)
                    if spread > best_neg_entropy:
                        best_neg_entropy = spread
                        best_alpha = a_candidate
                LOGGER.info(
                    "Validation-based alpha selection: best_alpha=%.2f "
                    "(spread=%.4f) from %d candidates.",
                    best_alpha,
                    best_neg_entropy,
                    len(alpha_candidates),
                )
                dual_alpha = best_alpha
                dual_scorer.alpha = dual_alpha
            else:
                LOGGER.warning(
                    "No validation scores in checkpoint for alpha selection; "
                    "using config alpha=%.2f. Re-run training to enable "
                    "validation-based selection.",
                    dual_alpha,
                )

            # Compute distributional scores from test tokens (target windows)
            k_val = int(cfg.patchtraj.K)
            delta_val = int(cfg.patchtraj.delta)
            seq_ds_len = len(sequence_dataset)
            start_idx = k_val + delta_val - 1
            target_all = token_windows[start_idx : start_idx + seq_ds_len]
            # Batch distributional scoring to limit memory usage
            dist_batch_size = 4096
            dist_chunks: list[npt.NDArray[np.float64]] = []
            for ds_start in range(0, target_all.shape[0], dist_batch_size):
                ds_end = min(ds_start + dist_batch_size, target_all.shape[0])
                dist_chunks.append(
                    dual_scorer.score_distributional(target_all[ds_start:ds_end])
                )
            dist_scores = np.concatenate(dist_chunks, axis=0)
            # Save separate scores for post-hoc ablation
            np.save(output_dir / "traj_scores.npy", raw_scores.astype(np.float64))
            np.save(output_dir / "dist_scores.npy", dist_scores.astype(np.float64))
            # If the checkpoint did not store reference normalizers (older
            # training runs), fit them from the cached validation scores so the
            # fuse() z-score is leak-free at test time.
            if (
                val_traj is not None
                and val_dist is not None
                and dual_scorer._traj_ref_mu is None  # pylint: disable=protected-access
            ):
                dual_scorer.fit_normalizers(
                    np.asarray(val_traj, dtype=np.float64),
                    np.asarray(val_dist, dtype=np.float64),
                )
            raw_scores = dual_scorer.fuse(
                traj_scores=raw_scores.astype(np.float64),
                dist_scores=dist_scores.astype(np.float64),
            )
            LOGGER.info(
                "Applied dual-signal scoring (alpha=%.2f).", dual_alpha
            )
        else:
            LOGGER.warning(
                "dual_signal.enabled=true but no dual_signal_state in checkpoint; "
                "falling back to trajectory-only scoring."
            )

    # Smooth scores to reduce noise (improves AUC-ROC)
    smooth_window = int(cfg.scoring.get("smooth_window", 5)) if "scoring" in cfg else 5
    smooth_method = (
        str(cfg.scoring.get("smooth_method", "mean")) if "scoring" in cfg else "mean"
    )
    if smooth_window > 1:
        if smooth_window % 2 == 0:
            smooth_window += 1
        raw_scores = smooth_scores(
            raw_scores, window_size=smooth_window, method=smooth_method
        )
        LOGGER.info(
            "Applied score smoothing: method=%s, window_size=%d",
            smooth_method,
            smooth_window,
        )

    # ------------------------------------------------------------------
    # Build a validation-side score stream from the training-split
    # validation cached in the checkpoint. Used for:
    #   1. fitting the min-max scaler (so test distribution is not used)
    #   2. selecting a leakage-free F1 / F1-PA threshold via quantile
    # ------------------------------------------------------------------
    val_smoothed: npt.NDArray[np.float64] | None = None
    dual_scorer_local = locals().get("dual_scorer", None)
    if isinstance(checkpoint, dict):
        val_traj_raw = checkpoint.get("val_traj_scores")
        val_dist_raw = checkpoint.get("val_dist_scores")
        if val_traj_raw is not None:
            val_traj_arr = np.asarray(val_traj_raw, dtype=np.float64).reshape(-1)
            if (
                bool(dual_cfg.get("enabled", False))
                and val_dist_raw is not None
                and dual_scorer_local is not None
            ):
                val_dist_arr = np.asarray(val_dist_raw, dtype=np.float64).reshape(-1)
                # Reuse the test-time scorer; fuse() uses its fitted state.
                val_fused = dual_scorer_local.fuse(
                    traj_scores=val_traj_arr,
                    dist_scores=val_dist_arr,
                )
            else:
                val_fused = val_traj_arr
            if smooth_window > 1:
                val_fused = smooth_scores(
                    val_fused.astype(np.float64),
                    window_size=smooth_window,
                    method=smooth_method,
                )
            val_smoothed = val_fused.astype(np.float64)

    # Fit a min-max scaler on validation scores when available so the
    # normalization reported for the test stream does NOT depend on test
    # distribution. Falls back to test-fitted minmax for backward compat
    # but logs a warning so the gap is visible.
    if val_smoothed is not None and val_smoothed.size > 0:
        smin = float(val_smoothed.min())
        smax = float(val_smoothed.max())
        denom = smax - smin
        if denom <= 0:
            LOGGER.warning(
                "Validation score range degenerate (min==max); falling back "
                "to test-fitted min-max normalization."
            )
            normalized_scores = normalize_scores(raw_scores, method="minmax")
            val_normalized = normalize_scores(val_smoothed, method="minmax")
            score_scaler_fit_split = "test (degenerate val)"
        else:
            normalized_scores = (raw_scores - smin) / denom
            val_normalized = (val_smoothed - smin) / denom
            score_scaler_fit_split = "validation (train-split)"
            LOGGER.info(
                "Min-max scaler fit on validation: smin=%.6g, smax=%.6g",
                smin, smax,
            )
    else:
        LOGGER.warning(
            "No validation scores in checkpoint; falling back to test-fitted "
            "min-max normalization. F1/F1-PA validation metrics will be NaN."
        )
        normalized_scores = normalize_scores(raw_scores, method="minmax")
        val_normalized = None
        score_scaler_fit_split = "test (no val available)"

    # Compute metrics: AUC-ROC/PR are scale-invariant; F1/F1-PA are split
    # into oracle (test-label-tuned, upper bound) and validation
    # (val-quantile threshold, deployable).
    # F1 / F1-PA validation threshold uses its own FAR target (default 5%).
    # Do NOT reuse `cfg.scoring.alpha`, which is the CalibGuard FAR (often
    # 0.01 — too strict for headline F1 reporting).
    far_target = (
        float(cfg.scoring.get("f1_far_target", 0.05))
        if "scoring" in cfg
        else 0.05
    )
    metrics = compute_all_metrics(
        normalized_scores,
        eval_labels,
        val_scores=val_normalized,
        far_target=far_target,
    )
    metrics["score_scaler_fit_split"] = score_scaler_fit_split
    metrics["smooth_window"] = smooth_window
    metrics["smooth_method"] = smooth_method
    metrics["far_target"] = far_target

    np.save(output_dir / "scores.npy", normalized_scores)
    np.save(output_dir / "labels.npy", eval_labels)
    if val_normalized is not None:
        np.save(output_dir / "val_scores.npy", val_normalized)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=str)

    LOGGER.info("Detection completed. Output directory: %s", output_dir)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
            LOGGER.info("%s=%.6f", metric_name, metric_value)
        else:
            LOGGER.info("%s=%s", metric_name, metric_value)

    # --- CalibGuard FAR evaluation (optional) ---
    calibguard_cfg = cfg.get("calibguard", None)
    if calibguard_cfg is not None and bool(calibguard_cfg.get("enabled", False)):
        _run_calibguard(
            scores=normalized_scores,
            labels=eval_labels,
            cfg=calibguard_cfg,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_apply_multiscale_cli_flag(sys.argv[1:])]
    main()  # pyright: ignore[reportCallIssue]
