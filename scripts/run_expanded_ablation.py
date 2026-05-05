from __future__ import annotations

# pyright: reportMissingImports=false,reportUnknownVariableType=false,reportUnknownMemberType=false,reportUnknownArgumentType=false,reportUnknownParameterType=false,reportUntypedBaseClass=false,reportMissingSuperCall=false,reportImplicitStringConcatenation=false

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.base import create_sliding_windows, normalize_data
from src.data.smd import _load_smd_labels, _load_smd_matrix
from src.evaluation.metrics import compute_all_metrics
from src.models.backbone import VisionBackbone
from src.models.patchtraj import PatchTrajPredictor
from src.models.tcn_ae import TCNAutoencoder
from src.rendering.line_plot import render_line_plot_batch
from src.rendering.multi_view import compute_view_disagreement
from src.rendering.recurrence_plot import render_recurrence_plot_batch
from src.rendering.token_correspondence import (
    compute_correspondence_map,
    compute_identity_map,
)
from src.scoring.calibguard_v3 import CalibGuardV3
from src.scoring.hybrid_scorer import compute_hybrid_score
from src.scoring.patchtraj_scorer import (
    compute_patchtraj_score,
    normalize_scores,
    smooth_scores,
)
from src.scoring.score_fusion import fuse_scores
from src.utils.reproducibility import get_device, seed_everything

LOGGER = logging.getLogger(__name__)

RESULTS_ROOT = ROOT / "results"
PREFERRED_OUTPUT_ROOT = RESULTS_ROOT / "improved_v2" / "ablation"
FALLBACK_OUTPUT_ROOT = (
    ROOT
    / ".sisyphus"
    / "artifacts"
    / "task-13"
    / "results"
    / "improved_v2"
    / "ablation"
)
EVIDENCE_CSV = ROOT / ".sisyphus" / "evidence" / "task-13-ablation-results.csv"

SELECTED_ENTITIES = [
    "machine-1-1",
    "machine-1-5",
    "machine-2-2",
    "machine-2-8",
    "machine-3-2",
]
WINDOW_SIZES = [50, 100, 200]
STRIDE = 10
VIEW_WEIGHTS = {"line_plot": 0.4, "recurrence_plot": 0.6}
VIEW_LAMBDA = 1.0
MULTISCALE_METHOD = "zscore_weighted"
VIEW_SMOOTH_WINDOW = 7
VIEW_SMOOTH_METHOD = "median"
DETECT_SMOOTH_WINDOW = 21
DETECT_SMOOTH_METHOD = "mean"
HYBRID_WEIGHT = 0.5
CALIB_ALPHA = 0.05
CALIB_RATIO = 0.2
TCN_EPOCHS = 8
TCN_BATCH_SIZE = 64
TCN_LR = 1e-3

RENDER_BACKBONE_ROUTE = {
    "line_plot": "clip",
    "recurrence_plot": "dinov2",
}

RENDERER_KWARGS: dict[str, dict[str, Any]] = {
    "line_plot": {
        "image_size": 224,
        "dpi": 100,
        "colormap": "tab10",
        "line_width": 1.0,
        "background_color": "white",
        "show_axes": False,
        "show_grid": False,
    },
    "recurrence_plot": {
        "image_size": 224,
        "metric": "euclidean",
    },
}

BACKBONE_MODELS = {
    "clip": "openai/clip-vit-base-patch16",
    "dinov2": "facebook/dinov2-base",
}

AXIS_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]
AXIS_LABELS = {
    "A": "pi_identity",
    "B": "pi_ot",
    "C": "cls_only",
    "D": "single_backbone",
    "E": "no_multiscale",
    "F": "no_viewdisagree",
    "G": "no_1d_branch",
    "H": "fixed_threshold",
}


@dataclass(frozen=True)
class EntitySeries:
    train_data: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray


@dataclass(frozen=True)
class WindowSplit:
    windows: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ScoreStream:
    scores: np.ndarray
    labels: np.ndarray


class TokenSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):

    def __init__(
        self,
        token_windows: np.ndarray,
        window_labels: np.ndarray,
        k: int,
        delta: int,
    ) -> None:
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
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self._k
        target_idx = end + self._delta - 1
        return (
            torch.from_numpy(self._tokens[start:end]),
            torch.from_numpy(self._tokens[target_idx]),
            torch.tensor(self._labels[target_idx], dtype=torch.int64),
        )


class ExpandedAblationRunner:

    def __init__(self, force: bool = False) -> None:
        self.force = force
        self.device = get_device()
        self.series_cache: dict[str, EntitySeries] = {}
        self.window_cache: dict[tuple[str, str, int], WindowSplit] = {}
        self.backbone_cache: dict[str, VisionBackbone] = {}
        self.predictor_cache: dict[tuple[str, str], PatchTrajPredictor] = {}
        self.output_root = self._resolve_output_root()
        self.cache_root = self.output_root / "_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _resolve_output_root(self) -> Path:
        try:
            PREFERRED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return PREFERRED_OUTPUT_ROOT
        except PermissionError:
            FALLBACK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                "Preferred output path %s is not writable; using fallback %s",
                PREFERRED_OUTPUT_ROOT,
                FALLBACK_OUTPUT_ROOT,
            )
            return FALLBACK_OUTPUT_ROOT

    def run(self, entities: list[str], axes: list[str]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        axis_summaries: dict[str, dict[str, Any]] = {}

        for entity in entities:
            LOGGER.info("Running expanded ablation for %s", entity)
            entity_rows = self._run_entity(entity=entity, axes=axes)
            rows.extend(entity_rows)

        for axis in axes:
            axis_rows = [row for row in rows if row["axis"] == axis]
            if not axis_rows:
                continue
            primary_key = axis_rows[0]["primary_metric"]
            primary_deltas = [float(row["primary_delta"]) for row in axis_rows]
            axis_summaries[axis] = {
                "axis": axis,
                "label": AXIS_LABELS[axis],
                "entities": len(axis_rows),
                "reference": axis_rows[0]["reference_name"],
                "variant": axis_rows[0]["variant_name"],
                "primary_metric": primary_key,
                "mean_primary_delta": float(np.mean(primary_deltas)),
                "median_primary_delta": float(np.median(primary_deltas)),
                "mean_auc_roc_delta": float(
                    np.mean([float(row["delta_auc_roc"]) for row in axis_rows])
                ),
                "mean_auc_pr_delta": float(
                    np.mean([float(row["delta_auc_pr"]) for row in axis_rows])
                ),
                "mean_f1_pa_delta": float(
                    np.mean([float(row["delta_f1_pa"]) for row in axis_rows])
                ),
            }

        summary = {
            "requested_output_root": str(PREFERRED_OUTPUT_ROOT),
            "actual_output_root": str(self.output_root),
            "entities": entities,
            "axes": axes,
            "rows": rows,
            "axis_summaries": axis_summaries,
        }
        self._write_outputs(summary)
        return summary

    def _run_entity(self, entity: str, axes: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        line_geo_clip = self._score_stream(
            entity=entity,
            split="test",
            renderer="line_plot",
            backbone_name="clip",
            token_mode="patch",
            correspondence="geometric",
            window_size=100,
        )
        recur_geo_dino = self._score_stream(
            entity=entity,
            split="test",
            renderer="recurrence_plot",
            backbone_name="dinov2",
            token_mode="patch",
            correspondence="geometric",
            window_size=100,
        )
        routed_geo = self._combine_views(line_geo_clip, recur_geo_dino, lambda_disagree=VIEW_LAMBDA)
        routed_geo_no_disagree = self._combine_views(
            line_geo_clip,
            recur_geo_dino,
            lambda_disagree=0.0,
        )

        if "A" in axes:
            line_identity = self._score_stream(
                entity=entity,
                split="test",
                renderer="line_plot",
                backbone_name="clip",
                token_mode="patch",
                correspondence="identity",
                window_size=100,
            )
            recur_identity = self._score_stream(
                entity=entity,
                split="test",
                renderer="recurrence_plot",
                backbone_name="dinov2",
                token_mode="patch",
                correspondence="identity",
                window_size=100,
            )
            rows.append(
                self._compare_streams(
                    axis="A",
                    entity=entity,
                    reference_name="geometric_patch",
                    reference=routed_geo,
                    variant_name="identity_patch",
                    variant=self._combine_views(
                        line_identity,
                        recur_identity,
                        lambda_disagree=VIEW_LAMBDA,
                    ),
                )
            )

        if "B" in axes:
            line_ot = self._score_stream(
                entity=entity,
                split="test",
                renderer="line_plot",
                backbone_name="clip",
                token_mode="patch",
                correspondence="ot",
                window_size=100,
            )
            recur_ot = self._score_stream(
                entity=entity,
                split="test",
                renderer="recurrence_plot",
                backbone_name="dinov2",
                token_mode="patch",
                correspondence="ot",
                window_size=100,
            )
            rows.append(
                self._compare_streams(
                    axis="B",
                    entity=entity,
                    reference_name="geometric_patch",
                    reference=routed_geo,
                    variant_name="ot_patch",
                    variant=self._combine_views(line_ot, recur_ot, lambda_disagree=VIEW_LAMBDA),
                )
            )

        if "C" in axes:
            line_cls = self._score_stream(
                entity=entity,
                split="test",
                renderer="line_plot",
                backbone_name="clip",
                token_mode="cls",
                correspondence="identity",
                window_size=100,
            )
            recur_cls = self._score_stream(
                entity=entity,
                split="test",
                renderer="recurrence_plot",
                backbone_name="dinov2",
                token_mode="cls",
                correspondence="identity",
                window_size=100,
            )
            rows.append(
                self._compare_streams(
                    axis="C",
                    entity=entity,
                    reference_name="patch_tokens",
                    reference=routed_geo,
                    variant_name="cls_only",
                    variant=self._combine_views(
                        line_cls,
                        recur_cls,
                        lambda_disagree=VIEW_LAMBDA,
                    ),
                )
            )

        if "D" in axes:
            line_geo_dino = self._score_stream(
                entity=entity,
                split="test",
                renderer="line_plot",
                backbone_name="dinov2",
                token_mode="patch",
                correspondence="geometric",
                window_size=100,
            )
            single_backbone = self._combine_views(
                line_geo_dino,
                recur_geo_dino,
                lambda_disagree=VIEW_LAMBDA,
            )
            rows.append(
                self._compare_streams(
                    axis="D",
                    entity=entity,
                    reference_name="routed_clip_lp_dino_rp",
                    reference=routed_geo,
                    variant_name="single_backbone_dinov2",
                    variant=single_backbone,
                )
            )

        if "E" in axes:
            per_scale: dict[str, ScoreStream] = {}
            for window_size in WINDOW_SIZES:
                scale_line = self._score_stream(
                    entity=entity,
                    split="test",
                    renderer="line_plot",
                    backbone_name="clip",
                    token_mode="patch",
                    correspondence="geometric",
                    window_size=window_size,
                )
                scale_recur = self._score_stream(
                    entity=entity,
                    split="test",
                    renderer="recurrence_plot",
                    backbone_name="dinov2",
                    token_mode="patch",
                    correspondence="geometric",
                    window_size=window_size,
                )
                per_scale[f"w{window_size}"] = self._combine_views(
                    scale_line,
                    scale_recur,
                    lambda_disagree=VIEW_LAMBDA,
                )
            rows.append(
                self._compare_streams(
                    axis="E",
                    entity=entity,
                    reference_name="multiscale_w50_w100_w200",
                    reference=self._combine_multiscale(per_scale),
                    variant_name="single_scale_w100",
                    variant=routed_geo,
                )
            )

        if "F" in axes:
            rows.append(
                self._compare_streams(
                    axis="F",
                    entity=entity,
                    reference_name="with_viewdisagree_lambda_1.0",
                    reference=routed_geo,
                    variant_name="without_viewdisagree_lambda_0.0",
                    variant=routed_geo_no_disagree,
                )
            )

        if "G" in axes:
            recon_stream = self._reconstruction_stream(entity=entity, window_size=100)
            hybrid_scores = compute_hybrid_score(
                patchtraj_score=routed_geo.scores,
                recon_score=recon_stream.scores,
                method="weighted_sum",
                weight=HYBRID_WEIGHT,
            )
            hybrid_scores = normalize_scores(hybrid_scores, method="minmax")
            hybrid_labels = routed_geo.labels[: hybrid_scores.shape[0]]
            hybrid_stream = ScoreStream(scores=hybrid_scores, labels=hybrid_labels)
            rows.append(
                self._compare_streams(
                    axis="G",
                    entity=entity,
                    reference_name="with_1d_branch",
                    reference=hybrid_stream,
                    variant_name="without_1d_branch",
                    variant=routed_geo,
                )
            )

        if "H" in axes:
            train_line = self._score_stream(
                entity=entity,
                split="train",
                renderer="line_plot",
                backbone_name="clip",
                token_mode="patch",
                correspondence="geometric",
                window_size=100,
            )
            train_recur = self._score_stream(
                entity=entity,
                split="train",
                renderer="recurrence_plot",
                backbone_name="dinov2",
                token_mode="patch",
                correspondence="geometric",
                window_size=100,
            )
            train_routed = self._combine_views(
                train_line,
                train_recur,
                lambda_disagree=VIEW_LAMBDA,
            )
            fixed_metrics = self._evaluate_fixed_threshold(
                train_scores=train_routed.scores,
                test_stream=routed_geo,
                alpha=CALIB_ALPHA,
            )
            calibguard_metrics = self._evaluate_calibguard_v3(
                train_scores=train_routed.scores,
                test_stream=routed_geo,
                alpha=CALIB_ALPHA,
            )
            rows.append(
                self._compare_calibration(
                    entity=entity,
                    fixed_metrics=fixed_metrics,
                    calibguard_metrics=calibguard_metrics,
                )
            )

        return rows

    def _load_entity_series(self, entity: str) -> EntitySeries:
        cached = self.series_cache.get(entity)
        if cached is not None:
            return cached

        raw_dir = ROOT / "data" / "raw" / "smd"
        train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
        test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
        test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")
        train_data, test_data = normalize_data(
            train_data=train_data,
            test_data=test_data,
            method="standard",
        )
        series = EntitySeries(
            train_data=train_data.astype(np.float64, copy=False),
            test_data=test_data.astype(np.float64, copy=False),
            test_labels=test_labels.astype(np.int64, copy=False),
        )
        self.series_cache[entity] = series
        return series

    def _window_split(self, entity: str, split: str, window_size: int) -> WindowSplit:
        key = (entity, split, window_size)
        cached = self.window_cache.get(key)
        if cached is not None:
            return cached

        series = self._load_entity_series(entity)
        if split == "train":
            data = series.train_data
            labels = np.zeros((series.train_data.shape[0],), dtype=np.int64)
        elif split == "test":
            data = series.test_data
            labels = series.test_labels
        else:
            raise ValueError(f"Unsupported split '{split}'.")

        windows, window_labels = create_sliding_windows(
            data=data,
            labels=labels,
            window_size=window_size,
            stride=STRIDE,
        )
        split_data = WindowSplit(windows=windows.astype(np.float32), labels=window_labels)
        self.window_cache[key] = split_data
        return split_data

    def _get_backbone(self, backbone_name: str) -> VisionBackbone:
        cached = self.backbone_cache.get(backbone_name)
        if cached is not None:
            return cached
        model_name = BACKBONE_MODELS[backbone_name]
        backbone = VisionBackbone(model_name=model_name, device=self.device)
        self.backbone_cache[backbone_name] = backbone
        return backbone

    def _get_predictor(self, entity: str, renderer: str, hidden_dim: int) -> PatchTrajPredictor:
        key = (entity, renderer)
        cached = self.predictor_cache.get(key)
        if cached is not None:
            return cached

        checkpoint_path = RESULTS_ROOT / "improved_smd" / entity / renderer / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = PatchTrajPredictor(
            hidden_dim=hidden_dim,
            d_model=384,
            n_heads=6,
            n_layers=3,
            dim_feedforward=1536,
            dropout=0.1,
            activation="gelu",
        ).to(self.device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
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
        self.predictor_cache[key] = model
        return model

    def _renderer_fn(self, renderer: str) -> Any:
        if renderer == "line_plot":
            return render_line_plot_batch
        if renderer == "recurrence_plot":
            return render_recurrence_plot_batch
        raise ValueError(f"Unsupported renderer '{renderer}'.")

    def _token_cache_base(
        self,
        entity: str,
        split: str,
        renderer: str,
        backbone_name: str,
        token_mode: str,
        window_size: int,
    ) -> Path:
        return (
            self.cache_root
            / entity
            / split
            / f"w{window_size}"
            / f"{renderer}_{backbone_name}_{token_mode}"
        )

    def _extract_tokens(
        self,
        entity: str,
        split: str,
        renderer: str,
        backbone_name: str,
        token_mode: str,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
        cache_base = self._token_cache_base(
            entity=entity,
            split=split,
            renderer=renderer,
            backbone_name=backbone_name,
            token_mode=token_mode,
            window_size=window_size,
        )
        tokens_path = cache_base / "tokens.npy"
        labels_path = cache_base / "labels.npy"
        meta_path = cache_base / "meta.json"
        if not self.force and tokens_path.exists() and labels_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            patch_grid = tuple(meta["patch_grid"]) if meta["patch_grid"] is not None else None
            return (
                np.load(tokens_path).astype(np.float32, copy=False),
                np.load(labels_path).astype(np.int64, copy=False),
                patch_grid,
            )

        split_data = self._window_split(entity=entity, split=split, window_size=window_size)
        backbone = self._get_backbone(backbone_name)
        render_fn = self._renderer_fn(renderer)
        cache_base.mkdir(parents=True, exist_ok=True)

        token_chunks: list[np.ndarray] = []
        batch_size = 32
        for start in range(0, split_data.windows.shape[0], batch_size):
            end = min(start + batch_size, split_data.windows.shape[0])
            images = render_fn(split_data.windows[start:end], **RENDERER_KWARGS[renderer])
            if token_mode == "patch":
                token_chunks.append(backbone.extract_patch_tokens_from_numpy(images))
            elif token_mode == "cls":
                token_chunks.append(self._extract_cls_tokens_from_numpy(backbone, images))
            else:
                raise ValueError(f"Unsupported token_mode '{token_mode}'.")

        tokens = np.concatenate(token_chunks, axis=0).astype(np.float32, copy=False)
        patch_grid = backbone.patch_grid if token_mode == "patch" else None
        np.save(tokens_path, tokens)
        np.save(labels_path, split_data.labels.astype(np.int64, copy=False))
        meta_path.write_text(
            json.dumps(
                {
                    "entity": entity,
                    "split": split,
                    "renderer": renderer,
                    "backbone": backbone_name,
                    "token_mode": token_mode,
                    "window_size": window_size,
                    "patch_grid": list(patch_grid) if patch_grid is not None else None,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return tokens, split_data.labels.astype(np.int64, copy=False), patch_grid

    def _extract_cls_tokens_from_numpy(
        self,
        backbone: VisionBackbone,
        images: np.ndarray,
    ) -> np.ndarray:
        image_tensor = torch.from_numpy(images)
        backbone._validate_images(image_tensor)
        pixel_values = image_tensor.to(
            device=backbone.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        pixel_values = (pixel_values - backbone._mean) / backbone._std
        pixel_values = pixel_values.to(dtype=backbone.dtype)
        with torch.no_grad():
            outputs = backbone.model(pixel_values=pixel_values, output_attentions=False)
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 1:
            raise RuntimeError(f"Unexpected token shape for CLS extraction: {tokens.shape}.")
        return tokens[:, :1, :].detach().cpu().numpy().astype(np.float32, copy=False)

    def _score_stream(
        self,
        entity: str,
        split: str,
        renderer: str,
        backbone_name: str,
        token_mode: str,
        correspondence: str,
        window_size: int,
    ) -> ScoreStream:
        result_dir = (
            self.cache_root
            / entity
            / split
            / f"w{window_size}"
            / f"{renderer}_{backbone_name}_{token_mode}_{correspondence}"
        )
        scores_path = result_dir / "scores.npy"
        labels_path = result_dir / "labels.npy"
        if not self.force and scores_path.exists() and labels_path.exists():
            return ScoreStream(
                scores=np.load(scores_path).astype(np.float64, copy=False),
                labels=np.load(labels_path).astype(np.int64, copy=False),
            )

        if (
            not self.force
            and split == "test"
            and window_size == 100
            and backbone_name == "dinov2"
            and token_mode == "patch"
            and correspondence == "geometric"
        ):
            existing_dir = RESULTS_ROOT / "improved_smd" / entity / renderer
            existing_scores = existing_dir / "scores.npy"
            existing_labels = existing_dir / "labels.npy"
            if existing_scores.exists() and existing_labels.exists():
                result_dir.mkdir(parents=True, exist_ok=True)
                scores = np.load(existing_scores).astype(np.float64, copy=False)
                labels = np.load(existing_labels).astype(np.int64, copy=False)
                np.save(scores_path, scores)
                np.save(labels_path, labels)
                return ScoreStream(scores=scores, labels=labels)

        tokens, window_labels, patch_grid = self._extract_tokens(
            entity=entity,
            split=split,
            renderer=renderer,
            backbone_name=backbone_name,
            token_mode=token_mode,
            window_size=window_size,
        )
        predictor = self._get_predictor(
            entity=entity,
            renderer=renderer,
            hidden_dim=int(tokens.shape[-1]),
        )
        pi, valid_mask = self._compute_correspondence(
            tokens=tokens,
            renderer=renderer,
            correspondence=correspondence,
            token_mode=token_mode,
            patch_grid=patch_grid,
            window_size=window_size,
        )

        dataset = TokenSequenceDataset(token_windows=tokens, window_labels=window_labels, k=12, delta=1)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        score_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for token_seq, actual_tokens, target_labels in dataloader:
                token_seq = token_seq.to(self.device, dtype=torch.float32, non_blocking=True)
                actual_tokens = actual_tokens.to(self.device, dtype=torch.float32, non_blocking=True)
                predicted_tokens = predictor(token_seq)
                batch_scores = compute_patchtraj_score(
                    predicted_tokens=predicted_tokens,
                    actual_tokens=actual_tokens,
                    pi=pi,
                    valid_mask=valid_mask,
                )
                score_chunks.append(batch_scores.detach().cpu().numpy().astype(np.float64))
                label_chunks.append(target_labels.numpy().astype(np.int64, copy=False))

        raw_scores = np.concatenate(score_chunks, axis=0)
        if DETECT_SMOOTH_WINDOW > 1:
            raw_scores = smooth_scores(
                raw_scores,
                window_size=DETECT_SMOOTH_WINDOW,
                method=DETECT_SMOOTH_METHOD,
            )
        normalized_scores = normalize_scores(raw_scores, method="minmax")
        eval_labels = np.concatenate(label_chunks, axis=0)

        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(scores_path, normalized_scores)
        np.save(labels_path, eval_labels)
        metrics_path = result_dir / "metrics.json"
        if split == "test" and np.unique(eval_labels).size > 1:
            metrics = compute_all_metrics(normalized_scores, eval_labels)
            metrics_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return ScoreStream(scores=normalized_scores, labels=eval_labels)

    def _compute_correspondence(
        self,
        tokens: np.ndarray,
        renderer: str,
        correspondence: str,
        token_mode: str,
        patch_grid: tuple[int, int] | None,
        window_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if token_mode == "cls":
            return np.array([0], dtype=np.int64), np.array([True], dtype=bool)

        if patch_grid is None:
            raise ValueError("patch_grid is required for patch-token correspondence.")

        if correspondence == "geometric":
            return compute_correspondence_map(
                renderer_type=renderer,
                window_size=window_size,
                stride=STRIDE,
                patch_grid=patch_grid,
            )
        if correspondence == "identity":
            return compute_identity_map(patch_grid=patch_grid)
        if correspondence == "ot":
            if tokens.shape[0] < 2:
                raise ValueError("OT correspondence requires at least two token windows.")
            hard_pi = self._sinkhorn_hard_assignment(
                torch.from_numpy(tokens[0]).to(self.device, dtype=torch.float32),
                torch.from_numpy(tokens[1]).to(self.device, dtype=torch.float32),
            )
            return hard_pi.detach().cpu().numpy().astype(np.int64, copy=False), np.ones(
                hard_pi.shape[0],
                dtype=bool,
            )
        raise ValueError(f"Unsupported correspondence '{correspondence}'.")

    def _sinkhorn_hard_assignment(
        self,
        tokens_t: torch.Tensor,
        tokens_t1: torch.Tensor,
        reg: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
    ) -> torch.Tensor:
        if tokens_t.ndim != 2 or tokens_t1.ndim != 2:
            raise ValueError("OT tokens must have shape (N, D).")
        if tokens_t.shape != tokens_t1.shape:
            raise ValueError(
                f"OT token shapes must match, got {tokens_t.shape} and {tokens_t1.shape}."
            )
        num_tokens = int(tokens_t.shape[0])
        a = torch.full((num_tokens,), 1.0 / num_tokens, device=tokens_t.device)
        b = torch.full((num_tokens,), 1.0 / num_tokens, device=tokens_t.device)
        cost = torch.cdist(tokens_t, tokens_t1, p=2).pow(2)
        kernel = torch.exp(-cost / reg).clamp_min(1e-12)
        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(max_iterations):
            u_prev = u.clone()
            v_prev = v.clone()
            v = b / (kernel.transpose(0, 1) @ u).clamp_min(1e-12)
            u = a / (kernel @ v).clamp_min(1e-12)
            if max(
                float(torch.max(torch.abs(u - u_prev)).item()),
                float(torch.max(torch.abs(v - v_prev)).item()),
            ) <= tolerance:
                break

        plan = (u[:, None] * kernel) * v[None, :]
        return torch.argmax(plan, dim=1).to(dtype=torch.int64)

    def _align_streams(self, *streams: ScoreStream) -> list[ScoreStream]:
        min_length = min(stream.scores.shape[0] for stream in streams)
        aligned: list[ScoreStream] = []
        for stream in streams:
            aligned.append(
                ScoreStream(
                    scores=stream.scores[:min_length].astype(np.float64, copy=False),
                    labels=stream.labels[:min_length].astype(np.int64, copy=False),
                )
            )
        return aligned

    def _combine_views(
        self,
        line_stream: ScoreStream,
        recur_stream: ScoreStream,
        lambda_disagree: float,
    ) -> ScoreStream:
        line_stream, recur_stream = self._align_streams(line_stream, recur_stream)
        base = fuse_scores(
            {
                "line_plot": line_stream.scores,
                "recurrence_plot": recur_stream.scores,
            },
            method="zscore_weighted",
            weights=VIEW_WEIGHTS,
        )
        stacked = np.stack(
            [
                self._zscore(line_stream.scores),
                self._zscore(recur_stream.scores),
            ],
            axis=0,
        )
        combined = base + lambda_disagree * compute_view_disagreement(stacked)
        if VIEW_SMOOTH_WINDOW > 1:
            combined = smooth_scores(
                combined,
                window_size=VIEW_SMOOTH_WINDOW,
                method=VIEW_SMOOTH_METHOD,
            )
        combined = normalize_scores(combined, method="minmax")
        return ScoreStream(scores=combined, labels=line_stream.labels)

    def _combine_multiscale(self, per_scale: dict[str, ScoreStream]) -> ScoreStream:
        ordered_keys = sorted(per_scale)
        min_length = min(per_scale[key].scores.shape[0] for key in ordered_keys)
        aligned_scores = {
            key: per_scale[key].scores[:min_length] for key in ordered_keys
        }
        labels = per_scale[ordered_keys[0]].labels[:min_length]
        combined = fuse_scores(aligned_scores, method=MULTISCALE_METHOD)
        if VIEW_SMOOTH_WINDOW > 1:
            combined = smooth_scores(
                combined,
                window_size=VIEW_SMOOTH_WINDOW,
                method=VIEW_SMOOTH_METHOD,
            )
        combined = normalize_scores(combined, method="minmax")
        return ScoreStream(scores=combined, labels=labels)

    def _reconstruction_stream(self, entity: str, window_size: int) -> ScoreStream:
        result_dir = self.cache_root / entity / "reconstruction" / f"w{window_size}"
        scores_path = result_dir / "scores.npy"
        labels_path = result_dir / "labels.npy"
        if not self.force and scores_path.exists() and labels_path.exists():
            return ScoreStream(
                scores=np.load(scores_path).astype(np.float64, copy=False),
                labels=np.load(labels_path).astype(np.int64, copy=False),
            )

        series = self._load_entity_series(entity)
        train_windows = self._window_split(entity=entity, split="train", window_size=window_size)
        test_windows = self._window_split(entity=entity, split="test", window_size=window_size)
        model = TCNAutoencoder(input_dim=int(series.train_data.shape[1]), dropout=0.0).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=TCN_LR)
        loss_fn = nn.MSELoss()
        train_tensor = torch.from_numpy(train_windows.windows).to(dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=TCN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        model.train()
        for _ in range(TCN_EPOCHS):
            for (batch_windows,) in train_loader:
                batch_windows = batch_windows.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                reconstructed = model(batch_windows)
                loss = loss_fn(reconstructed, batch_windows)
                loss.backward()
                optimizer.step()

        model.eval()
        test_tensor = torch.from_numpy(test_windows.windows).to(dtype=torch.float32)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, test_tensor.shape[0], TCN_BATCH_SIZE):
                end = min(start + TCN_BATCH_SIZE, test_tensor.shape[0])
                batch_windows = test_tensor[start:end].to(self.device, non_blocking=True)
                batch_scores = model.compute_reconstruction_score(batch_windows)
                scores.append(batch_scores.detach().cpu().numpy().astype(np.float64))

        recon_scores = np.concatenate(scores, axis=0)
        if recon_scores.shape[0] > 12:
            recon_scores = recon_scores[12:]
        recon_scores = normalize_scores(recon_scores, method="minmax")
        labels = test_windows.labels[12:12 + recon_scores.shape[0]].astype(np.int64, copy=False)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(scores_path, recon_scores)
        np.save(labels_path, labels)
        return ScoreStream(scores=recon_scores, labels=labels)

    def _evaluate_fixed_threshold(
        self,
        train_scores: np.ndarray,
        test_stream: ScoreStream,
        alpha: float,
    ) -> dict[str, float]:
        threshold = float(np.quantile(train_scores, 1.0 - alpha))
        flags = test_stream.scores > threshold
        normal_mask = test_stream.labels == 0
        anomaly_mask = test_stream.labels == 1
        actual_far = float(np.mean(flags[normal_mask])) if normal_mask.any() else 0.0
        recall = float(np.mean(flags[anomaly_mask])) if anomaly_mask.any() else 0.0
        return {
            "threshold": threshold,
            "actual_far": actual_far,
            "far_gap": actual_far - alpha,
            "recall": recall,
        }

    def _evaluate_calibguard_v3(
        self,
        train_scores: np.ndarray,
        test_stream: ScoreStream,
        alpha: float,
    ) -> dict[str, float]:
        guard = CalibGuardV3.from_train_split(
            train_scores=train_scores.astype(np.float64, copy=False),
            calib_ratio=CALIB_RATIO,
            alpha=alpha,
            rolling_window=0,
            use_aci=False,
        )
        flags, _, thresholds = guard.predict_batch(test_stream.scores.astype(np.float64, copy=False))
        normal_mask = test_stream.labels == 0
        anomaly_mask = test_stream.labels == 1
        actual_far = float(np.mean(flags[normal_mask])) if normal_mask.any() else 0.0
        recall = float(np.mean(flags[anomaly_mask])) if anomaly_mask.any() else 0.0
        return {
            "threshold": float(np.mean(thresholds)),
            "actual_far": actual_far,
            "far_gap": actual_far - alpha,
            "recall": recall,
        }

    def _compare_streams(
        self,
        axis: str,
        entity: str,
        reference_name: str,
        reference: ScoreStream,
        variant_name: str,
        variant: ScoreStream,
    ) -> dict[str, Any]:
        reference, variant = self._align_streams(reference, variant)
        reference_metrics = compute_all_metrics(reference.scores, reference.labels)
        variant_metrics = compute_all_metrics(variant.scores, variant.labels)
        return {
            "axis": axis,
            "axis_label": AXIS_LABELS[axis],
            "entity": entity,
            "reference_name": reference_name,
            "variant_name": variant_name,
            "reference_auc_roc": float(reference_metrics["auc_roc"]),
            "variant_auc_roc": float(variant_metrics["auc_roc"]),
            "delta_auc_roc": float(variant_metrics["auc_roc"] - reference_metrics["auc_roc"]),
            "reference_auc_pr": float(reference_metrics["auc_pr"]),
            "variant_auc_pr": float(variant_metrics["auc_pr"]),
            "delta_auc_pr": float(variant_metrics["auc_pr"] - reference_metrics["auc_pr"]),
            "reference_f1_pa": float(reference_metrics["f1_pa"]),
            "variant_f1_pa": float(variant_metrics["f1_pa"]),
            "delta_f1_pa": float(variant_metrics["f1_pa"] - reference_metrics["f1_pa"]),
            "primary_metric": "auc_roc",
            "primary_delta": float(variant_metrics["auc_roc"] - reference_metrics["auc_roc"]),
        }

    def _compare_calibration(
        self,
        entity: str,
        fixed_metrics: dict[str, float],
        calibguard_metrics: dict[str, float],
    ) -> dict[str, Any]:
        fixed_abs_gap = abs(fixed_metrics["far_gap"])
        calib_abs_gap = abs(calibguard_metrics["far_gap"])
        return {
            "axis": "H",
            "axis_label": AXIS_LABELS["H"],
            "entity": entity,
            "reference_name": "calibguard_v3",
            "variant_name": "fixed_quantile_threshold",
            "reference_auc_roc": np.nan,
            "variant_auc_roc": np.nan,
            "delta_auc_roc": np.nan,
            "reference_auc_pr": np.nan,
            "variant_auc_pr": np.nan,
            "delta_auc_pr": np.nan,
            "reference_f1_pa": calibguard_metrics["recall"],
            "variant_f1_pa": fixed_metrics["recall"],
            "delta_f1_pa": fixed_metrics["recall"] - calibguard_metrics["recall"],
            "reference_actual_far": calibguard_metrics["actual_far"],
            "variant_actual_far": fixed_metrics["actual_far"],
            "reference_far_gap": calibguard_metrics["far_gap"],
            "variant_far_gap": fixed_metrics["far_gap"],
            "delta_far_gap": fixed_metrics["far_gap"] - calibguard_metrics["far_gap"],
            "primary_metric": "abs_far_gap",
            "primary_delta": fixed_abs_gap - calib_abs_gap,
        }

    def _zscore(self, scores: np.ndarray) -> np.ndarray:
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std <= 1e-12:
            return np.zeros_like(scores, dtype=np.float64)
        return ((scores - mean) / std).astype(np.float64, copy=False)

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        rows = summary["rows"]
        csv_path = self.output_root / "expanded_ablation_results.csv"
        json_path = self.output_root / "expanded_ablation_results.json"
        md_path = self.output_root / "expanded_ablation_summary.md"
        selected_path = self.output_root / "selected_entities.json"

        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        selected_path.write_text(
            json.dumps(
                {
                    "entities": summary["entities"],
                    "selection_rationale": "machine-1-1 baseline + four additional SMD entities spanning strong and weak regimes across groups 1-3",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        md_path.write_text(self._build_markdown(summary), encoding="utf-8")

        EVIDENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_CSV.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _build_markdown(self, summary: dict[str, Any]) -> str:
        lines = ["# Expanded Ablation Summary", ""]
        lines.append(f"Entities: {', '.join(summary['entities'])}")
        lines.append("")
        lines.append("| Axis | Reference | Variant | Mean Primary Delta | Mean AUC-ROC Delta | Mean F1-PA Delta |")
        lines.append("|---|---|---|---:|---:|---:|")
        for axis in AXIS_ORDER:
            axis_summary = summary["axis_summaries"].get(axis)
            if axis_summary is None:
                continue
            lines.append(
                f"| {axis} ({axis_summary['label']}) | {axis_summary['reference']} | {axis_summary['variant']} | {axis_summary['mean_primary_delta']:+.4f} | {axis_summary['mean_auc_roc_delta']:+.4f} | {axis_summary['mean_f1_pa_delta']:+.4f} |"
            )
        lines.append("")
        lines.append("## Per-Entity Rows")
        lines.append("")
        lines.append("| Axis | Entity | Reference | Variant | Delta AUC-ROC | Delta AUC-PR | Delta F1-PA | Primary Delta |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for row in summary["rows"]:
            delta_auc_roc = row.get("delta_auc_roc")
            delta_auc_pr = row.get("delta_auc_pr")
            delta_f1_pa = row.get("delta_f1_pa")
            lines.append(
                f"| {row['axis']} | {row['entity']} | {row['reference_name']} | {row['variant_name']} | {self._fmt_metric(delta_auc_roc)} | {self._fmt_metric(delta_auc_pr)} | {self._fmt_metric(delta_f1_pa)} | {self._fmt_metric(row['primary_delta'])} |"
            )
        return "\n".join(lines) + "\n"

    def _fmt_metric(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float) and np.isnan(value):
            return "-"
        return f"{float(value):+.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded ablation study.")
    parser.add_argument(
        "--entities",
        nargs="*",
        default=SELECTED_ENTITIES,
        help="SMD entities to evaluate.",
    )
    parser.add_argument(
        "--axes",
        nargs="*",
        default=AXIS_ORDER,
        choices=AXIS_ORDER,
        help="Ablation axes to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached ablation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    seed_everything(42)
    runner = ExpandedAblationRunner(force=bool(args.force))
    summary = runner.run(entities=list(args.entities), axes=list(args.axes))
    LOGGER.info("Saved expanded ablation outputs to %s", runner.output_root)
    LOGGER.info(
        "Completed %d axis/entity comparisons.",
        len(summary["rows"]),
    )


if __name__ == "__main__":
    main()
