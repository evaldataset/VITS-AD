"""Train PatchTraj predictor with Hydra config."""

from __future__ import annotations

# pyright: basic, reportMissingImports=false

import logging
import math
from pathlib import Path
from typing import Any, Sequence, Tuple, cast

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.data.base import create_sliding_windows, normalize_data, time_based_split
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
    compute_patchtraj_residuals,
    compute_patchtraj_residuals_soft,
    compute_patchtraj_score,
    trimmed_huber_loss,
)
from src.scoring.dual_signal_scorer import DualSignalScorer
from src.utils.reproducibility import get_device, seed_everything


LOGGER = logging.getLogger(__name__)


class TokenSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Token sequence dataset for PatchTraj.

    Each sample is `(token_seq, target)` where `token_seq` has shape `(K, N, D)`
    and `target` has shape `(N, D)`.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        anchors: Sequence[int],
        k: int,
        delta: int,
    ) -> None:
        """Initialize sequence dataset.

        Args:
            tokens: Token tensor with shape `(T, N, D)`.
            anchors: Anchor indices `i` used to build `[i-K+1, ..., i]`.
            k: Sequence length.
            delta: Prediction horizon in windows.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (T, N, D), got {tokens.shape}.")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}.")
        if delta <= 0:
            raise ValueError(f"delta must be > 0, got {delta}.")

        self.tokens = tokens
        self.k = k
        self.delta = delta
        self.anchors = torch.as_tensor(list(anchors), dtype=torch.long)

        if self.anchors.numel() == 0:
            raise ValueError("anchors must be non-empty.")

    def __len__(self) -> int:
        return int(self.anchors.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = int(self.anchors[idx].item())
        token_seq = self.tokens[anchor - self.k + 1 : anchor + 1]
        target = self.tokens[anchor + self.delta]
        return token_seq, target


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _sanitize_component(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _extract_dataset_arrays(dataset: object) -> tuple[np.ndarray, np.ndarray] | None:
    data = getattr(dataset, "data", None)
    labels = getattr(dataset, "labels", None)
    if data is None or labels is None:
        return None

    data_array = np.asarray(data, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)

    if data_array.ndim != 2:
        raise ValueError(f"dataset.data must be 2D, got {data_array.shape}.")
    if labels_array.ndim != 1:
        raise ValueError(f"dataset.labels must be 1D, got {labels_array.shape}.")
    if data_array.shape[0] != labels_array.shape[0]:
        raise ValueError(
            "dataset.data and dataset.labels length mismatch: "
            f"{data_array.shape[0]} vs {labels_array.shape[0]}."
        )

    return data_array, (labels_array > 0).astype(np.int64, copy=False)


def _load_smd(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    from src.data.smd import SMDDataset, _load_smd_labels, _load_smd_matrix

    raw_dir = Path(to_absolute_path(str(cfg.data.raw_dir)))
    entity = str(cfg.data.entity)

    dataset = SMDDataset(
        raw_dir=raw_dir,
        entity=entity,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        normalize=bool(cfg.data.normalize),
        norm_method=str(cfg.data.norm_method),
    )

    extracted = _extract_dataset_arrays(dataset)
    if extracted is not None:
        LOGGER.info("Loaded SMD arrays from SMDDataset.data/.labels for %s.", entity)
        return extracted[0], extracted[1], entity, None

    LOGGER.info("Loading SMD %s via official train/test files.", entity)
    train_data = _load_smd_matrix(raw_dir / "train" / f"{entity}.txt")
    test_data = _load_smd_matrix(raw_dir / "test" / f"{entity}.txt")
    test_labels = _load_smd_labels(raw_dir / "test_label" / f"{entity}.txt")

    data = np.concatenate([train_data, test_data], axis=0).astype(
        np.float64, copy=False
    )
    labels = np.concatenate(
        [
            np.zeros((train_data.shape[0],), dtype=np.int64),
            test_labels.astype(np.int64),
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    # Return official train/test boundary so the main loop uses the official
    # split rather than concatenated 50/50 time-based split.
    return data, labels, entity, int(train_data.shape[0])


def _load_psm(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    from src.data.psm import PSMDataset, _load_psm_features, _load_psm_labels

    raw_dir = Path(to_absolute_path(str(cfg.data.raw_dir)))
    entity = "psm"

    dataset = PSMDataset(
        raw_dir=raw_dir,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        normalize=bool(cfg.data.normalize),
        norm_method=str(cfg.data.norm_method),
    )

    extracted = _extract_dataset_arrays(dataset)
    if extracted is not None:
        LOGGER.info("Loaded PSM arrays from PSMDataset.data/.labels.")
        return extracted[0], extracted[1], entity, None

    LOGGER.info("Loading PSM via official train/test files.")
    train_data = _load_psm_features(raw_dir / "train.csv")
    test_data = _load_psm_features(raw_dir / "test.csv")
    test_labels = _load_psm_labels(raw_dir / "test_label.csv")

    data = np.concatenate([train_data, test_data], axis=0).astype(
        np.float64, copy=False
    )
    labels = np.concatenate(
        [
            np.zeros((train_data.shape[0],), dtype=np.int64),
            test_labels.astype(np.int64),
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    # Use official train/test boundary instead of concatenated 50/50 split.
    return data, labels, entity, int(train_data.shape[0])


def _load_msl(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    from src.data.msl import MSLDataset, _load_msl_labels, _load_msl_matrix

    raw_dir = Path(to_absolute_path(str(cfg.data.raw_dir)))
    entity = "msl"

    dataset = MSLDataset(
        raw_dir=raw_dir,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        normalize=bool(cfg.data.normalize),
        norm_method=str(cfg.data.norm_method),
    )

    extracted = _extract_dataset_arrays(dataset)
    if extracted is not None:
        LOGGER.info("Loaded MSL arrays from MSLDataset.data/.labels.")
        return extracted[0], extracted[1], entity, None

    LOGGER.warning("MSLDataset has no .data/.labels; using raw-file loaders.")
    train_data = _load_msl_matrix(raw_dir / "MSL_train.npy")
    test_data = _load_msl_matrix(raw_dir / "MSL_test.npy")
    test_labels = _load_msl_labels(raw_dir / "MSL_test_label.npy")

    data = np.concatenate([train_data, test_data], axis=0).astype(
        np.float64, copy=False
    )
    labels = np.concatenate(
        [
            np.zeros((train_data.shape[0],), dtype=np.int64),
            test_labels.astype(np.int64),
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    return data, labels, entity, int(train_data.shape[0])


def _load_smap(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    from src.data.smap import SMAPDataset, _load_smap_labels, _load_smap_matrix

    raw_dir = Path(to_absolute_path(str(cfg.data.raw_dir)))
    entity = "smap"

    dataset = SMAPDataset(
        raw_dir=raw_dir,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        normalize=bool(cfg.data.normalize),
        norm_method=str(cfg.data.norm_method),
    )

    extracted = _extract_dataset_arrays(dataset)
    if extracted is not None:
        LOGGER.info("Loaded SMAP arrays from SMAPDataset.data/.labels.")
        return extracted[0], extracted[1], entity, None

    LOGGER.warning("SMAPDataset has no .data/.labels; using raw-file loaders.")
    train_data = _load_smap_matrix(raw_dir / "SMAP_train.npy")
    test_data = _load_smap_matrix(raw_dir / "SMAP_test.npy")
    test_labels = _load_smap_labels(raw_dir / "SMAP_test_label.npy")

    data = np.concatenate([train_data, test_data], axis=0).astype(
        np.float64, copy=False
    )
    labels = np.concatenate(
        [
            np.zeros((train_data.shape[0],), dtype=np.int64),
            test_labels.astype(np.int64),
        ],
        axis=0,
    ).astype(np.int64, copy=False)
    return data, labels, entity, int(train_data.shape[0])


def load_dataset_arrays(
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    """Load full time-series arrays from configured dataset.

    Args:
        cfg: Hydra config.

    Returns:
        ``(data, labels, entity, official_test_start)`` where ``data`` is ``(T, D)``,
        ``labels`` is ``(T,)``, and ``official_test_start`` is the index where the
        official test set begins (``None`` when the dataset uses ``train_ratio``).
    """
    name = str(cfg.data.name).strip().lower()
    if name == "smd":
        return _load_smd(cfg)
    if name == "psm":
        return _load_psm(cfg)
    if name == "msl":
        return _load_msl(cfg)
    if name == "smap":
        return _load_smap(cfg)
    raise ValueError(
        f"Unsupported dataset '{cfg.data.name}'. Expected ['smd', 'psm', 'msl', 'smap']."
    )


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
    raise ValueError(
        f"Unsupported render.name '{name}'. Expected one of ['line_plot', 'gaf', 'recurrence_plot']."
    )


def _infer_patch_grid(tokens: torch.Tensor) -> tuple[int, int]:
    num_patches = int(tokens.shape[1])
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot infer square patch grid from N={num_patches}.")
    return side, side


def _get_batch_renderer(name: str) -> Any:
    """Get batch renderer function by name."""
    if name == "line_plot":
        return render_line_plot_batch
    if name == "gaf":
        return render_gaf_batch
    if name == "recurrence_plot":
        return render_recurrence_plot_batch
    raise ValueError(
        f"Unsupported render.name '{name}'. Expected one of ['line_plot', 'gaf', 'recurrence_plot']."
    )


@torch.no_grad()
def extract_tokens_from_windows(
    windows: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Render windows and extract frozen backbone patch tokens.

    Supports line_plot, gaf, and recurrence_plot renderers.

    Args:
        windows: Window tensor with shape `(B, L, D)`.
        cfg: Hydra config.
        device: Device for backbone inference.

    Returns:
        `(tokens, patch_grid)` where `tokens` has shape `(B, N, H)`.
    """
    renderer_name = str(cfg.render.name)
    batch_render_fn = _get_batch_renderer(renderer_name)
    render_kwargs = _render_kwargs(cfg)

    backbone = VisionBackbone(model_name=str(cfg.model.pretrained), device=device)
    batch_size = int(cfg.training.batch_size)
    token_chunks: list[torch.Tensor] = []

    feature_layers_cfg = cfg.model.get("feature_layers", None)
    use_multilayer = feature_layers_cfg is not None and len(feature_layers_cfg) > 0
    if use_multilayer:
        feature_layers = tuple(int(idx) for idx in feature_layers_cfg)
        LOGGER.info("Using multi-layer features: layers=%s", feature_layers)

    for start in range(0, int(windows.shape[0]), batch_size):
        end = min(start + batch_size, int(windows.shape[0]))
        images = batch_render_fn(windows[start:end], **render_kwargs)
        image_tensor = torch.from_numpy(images.astype(np.float32, copy=False))
        if use_multilayer:
            batch_tokens = backbone.extract_multilayer_tokens(
                image_tensor, layers=feature_layers
            )
        else:
            batch_tokens = backbone.extract_patch_tokens(image_tensor)
        token_chunks.append(batch_tokens.detach().cpu().to(torch.float32))

    return torch.cat(token_chunks, dim=0), backbone.patch_grid


def load_or_cache_tokens(
    windows: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    entity: str,
) -> tuple[torch.Tensor, tuple[int, int], Path]:
    """Load cached tokens or extract and cache them.

    Args:
        windows: Training windows `(B, L, D)`.
        cfg: Hydra config.
        device: Device for backbone inference.
        entity: Dataset entity id.

    Returns:
        `(tokens, patch_grid, cache_path)`.
    """
    processed_dir = Path(to_absolute_path(str(cfg.data.processed_dir)))
    processed_dir.mkdir(parents=True, exist_ok=True)

    model_key = _sanitize_component(str(cfg.model.pretrained))
    entity_key = _sanitize_component(entity)
    render_key = _sanitize_component(str(cfg.render.name))
    cache_path = processed_dir / f"tokens_{model_key}_{entity_key}_{render_key}.pt"

    expected_meta = {
        "dataset": str(cfg.data.name),
        "entity": entity,
        "model": str(cfg.model.pretrained),
        "render": str(cfg.render.name),
        "window_size": int(cfg.data.window_size),
        "stride": int(cfg.data.stride),
        "image_size": int(cfg.render.image_size),
        "normalize": bool(cfg.data.normalize),
        "norm_method": str(cfg.data.norm_method),
    }

    if cache_path.exists():
        LOGGER.info("Loading token cache: %s", cache_path)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)  # nosec: trusted local checkpoint

        if isinstance(payload, dict) and "tokens" in payload:
            cached_meta = payload.get("meta")
            if cached_meta is None or cached_meta == expected_meta:
                tokens = torch.as_tensor(payload["tokens"], dtype=torch.float32)
                patch_grid = payload.get("patch_grid")
                if patch_grid is None:
                    patch_grid = _infer_patch_grid(tokens)
                return tokens, (int(patch_grid[0]), int(patch_grid[1])), cache_path
            LOGGER.warning("Token cache metadata mismatch. Recomputing cache.")

        elif isinstance(payload, torch.Tensor):
            payload_tensor = cast(torch.Tensor, payload)
            tokens = payload_tensor.to(dtype=torch.float32, device=torch.device("cpu"))
            return tokens, _infer_patch_grid(tokens), cache_path

        else:
            LOGGER.warning("Unsupported cache format. Recomputing cache.")

    LOGGER.info("Extracting tokens for %d windows.", int(windows.shape[0]))
    tokens, patch_grid = extract_tokens_from_windows(
        windows=windows, cfg=cfg, device=device
    )
    torch.save(
        {"tokens": tokens, "patch_grid": patch_grid, "meta": expected_meta}, cache_path
    )
    LOGGER.info("Saved token cache to %s", cache_path)
    return tokens, patch_grid, cache_path


def build_datasets(
    tokens: torch.Tensor,
    k: int,
    delta: int,
    val_ratio: float = 0.2,
) -> tuple[TokenSequenceDataset, TokenSequenceDataset]:
    """Build train/validation token-sequence datasets.

    Args:
        tokens: Token tensor `(T, N, D)`.
        k: Sequence length.
        delta: Prediction horizon.
        val_ratio: Validation fraction from the end of timeline.

    Returns:
        `(train_dataset, val_dataset)`.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")

    total_windows = int(tokens.shape[0])
    start_anchor = k - 1
    end_anchor = total_windows - delta - 1
    if end_anchor < start_anchor:
        raise ValueError(
            "Not enough windows for sequence building: "
            f"T={total_windows}, K={k}, delta={delta}."
        )

    anchors = np.arange(start_anchor, end_anchor + 1, dtype=np.int64)
    num_samples = int(anchors.shape[0])
    num_val = max(1, int(num_samples * val_ratio))
    num_train = num_samples - num_val
    if num_train <= 0:
        raise ValueError(f"Validation split leaves no train samples: {num_samples}.")

    gap = k  # Temporal gap to prevent window overlap between train and val
    train_anchors = anchors[:num_train]
    val_anchors = anchors[num_train + gap:]  # Skip 'gap' anchors
    if len(val_anchors) == 0:
        LOGGER.warning(
            "Temporal gap (k=%d) leaves no val anchors; falling back to no-gap split.",
            k,
        )
        val_anchors = anchors[num_train:]

    train_ds = TokenSequenceDataset(tokens, train_anchors.tolist(), k, delta)
    val_ds = TokenSequenceDataset(tokens, val_anchors.tolist(), k, delta)
    return train_ds, val_ds


def build_scheduler(
    optimizer: Adam,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
) -> LambdaLR:
    """Build cosine scheduler with linear warmup."""
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(0, warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        current = step + 1
        if warmup_steps > 0 and current <= warmup_steps:
            return float(current) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def maybe_init_wandb(cfg: DictConfig, output_dir: Path) -> Any | None:
    """Initialize wandb run if available, else return None."""
    try:
        import wandb
    except Exception:
        LOGGER.warning("wandb not installed; continuing without wandb logging.")
        return None

    tags = [str(tag) for tag in cfg.wandb.tags] if cfg.wandb.tags is not None else None
    config_obj = OmegaConf.to_container(cfg, resolve=True)
    config_dict = (
        cast(dict[str, Any], config_obj) if isinstance(config_obj, dict) else None
    )

    kwargs: dict[str, Any] = {
        "project": str(cfg.wandb.project),
        "entity": cfg.wandb.entity,
        "tags": tags,
        "dir": str(output_dir),
    }
    if config_dict is not None:
        kwargs["config"] = config_dict

    try:
        return wandb.init(**kwargs)
    except Exception:
        LOGGER.exception("wandb.init failed; continuing without wandb logging.")
        return None


def train_one_epoch(
    model: PatchTrajPredictor,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: Adam,
    scheduler: LambdaLR,
    device: torch.device,
    pi: np.ndarray,
    valid_mask: np.ndarray,
    ot_soft_pi: torch.Tensor | None,
    huber_delta: float,
    trimmed_ratio: float,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    running = 0.0

    for token_seq, target in loader:
        token_seq = token_seq.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predicted = model(token_seq)
        if ot_soft_pi is None:
            residuals = compute_patchtraj_residuals(predicted, target, pi, valid_mask)
        else:
            residuals = compute_patchtraj_residuals_soft(predicted, target, ot_soft_pi)
        loss = trimmed_huber_loss(
            residuals, delta=huber_delta, trim_ratio=trimmed_ratio
        )
        loss.backward()
        _ = optimizer.step()
        _ = scheduler.step()
        running += float(loss.item())

    return running / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(
    model: PatchTrajPredictor,
    loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    pi: np.ndarray,
    valid_mask: np.ndarray,
    ot_soft_pi: torch.Tensor | None,
    huber_delta: float,
    trimmed_ratio: float,
) -> float:
    """Validate one epoch and return average loss."""
    model.eval()
    running = 0.0

    for token_seq, target in loader:
        token_seq = token_seq.to(device=device, dtype=torch.float32, non_blocking=True)
        target = target.to(device=device, dtype=torch.float32, non_blocking=True)

        predicted = model(token_seq)
        if ot_soft_pi is None:
            residuals = compute_patchtraj_residuals(predicted, target, pi, valid_mask)
        else:
            residuals = compute_patchtraj_residuals_soft(predicted, target, ot_soft_pi)
        loss = trimmed_huber_loss(
            residuals, delta=huber_delta, trim_ratio=trimmed_ratio
        )
        running += float(loss.item())

    return running / max(1, len(loader))


@hydra.main(
    config_path="../configs",
    config_name="experiment/patchtraj_default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run PatchTraj training pipeline."""
    configure_logging()
    seed_everything(int(cfg.training.seed))
    device = get_device(None)
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    data, labels, entity, official_test_start = load_dataset_arrays(cfg)
    if official_test_start is not None:
        # Official split datasets (MSL, SMAP): use exact boundary.
        train_ratio = float(official_test_start) / float(len(labels))
        LOGGER.info(
            "Using official split boundary: train_ratio=%.4f (idx=%d/%d).",
            train_ratio,
            official_test_start,
            len(labels),
        )
    else:
        train_ratio = float(cfg.data.train_ratio)
    train_data, train_labels, test_data, _ = time_based_split(
        data=data,
        labels=labels,
        train_ratio=train_ratio,
    )

    if bool(cfg.data.normalize):
        train_data, _ = normalize_data(
            train_data=train_data,
            test_data=test_data,
            method=str(cfg.data.norm_method),
        )

    windows, window_labels = create_sliding_windows(
        data=train_data,
        labels=train_labels,
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
    )
    windows = windows[window_labels == 0]

    tokens, patch_grid, cache_path = load_or_cache_tokens(
        windows=windows,
        cfg=cfg,
        device=device,
        entity=entity,
    )

    k = int(cfg.patchtraj.K)
    delta = int(cfg.patchtraj.delta)
    train_ds, val_ds = build_datasets(tokens=tokens, k=k, delta=delta, val_ratio=0.2)

    pin_memory = device.type == "cuda"
    # Deterministic data loading: torch.Generator drives the shuffle RNG with
    # a single seed (seed_everything has already seeded torch globally).  The
    # worker_init_fn keeps determinism if a future config bumps num_workers
    # above zero.
    loader_seed = int(cfg.training.seed)

    def _worker_init_fn(worker_id: int) -> None:
        np.random.seed(loader_seed + worker_id)
        torch.manual_seed(loader_seed + worker_id)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(loader_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        generator=loader_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )

    use_identity_pi = bool(cfg.patchtraj.get("use_identity_pi", False))
    ot_soft_pi: torch.Tensor | None = None
    correspondence_mode = str(
        cfg.patchtraj.get("correspondence_mode", "geometric")
    ).strip().lower()
    if use_identity_pi:
        LOGGER.info("Using identity correspondence map (ablation mode)")
        pi, valid_mask = compute_identity_map(patch_grid=patch_grid)
    elif correspondence_mode == "ot":
        if int(tokens.shape[0]) < 2:
            raise ValueError(
                "OT correspondence requires at least two windows of tokens."
            )
        soft_pi, sinkhorn_iterations = compute_ot_correspondence(
            tokens_t=tokens[0],
            tokens_t1=tokens[1],
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
        ot_soft_pi = soft_pi
        valid_mask = np.ones_like(pi, dtype=bool)
        LOGGER.info(
            "Using OT correspondence map (reg=%.4f, iterations=%d)",
            float(cfg.patchtraj.get("ot_reg", 0.1)),
            sinkhorn_iterations,
        )
    elif correspondence_mode == "geometric":
        pi, valid_mask = compute_correspondence_map(
            renderer_type=str(cfg.render.name),
            window_size=int(cfg.data.window_size),
            stride=int(cfg.data.stride),
            patch_grid=patch_grid,
        )
    else:
        raise ValueError(
            "patchtraj.correspondence_mode must be one of "
            f"['geometric', 'ot'], got '{correspondence_mode}'."
        )

    use_spatial = bool(cfg.patchtraj.get("spatial_attention", False))
    if use_spatial:
        model: nn.Module = SpatialTemporalPatchTrajPredictor(
            hidden_dim=int(tokens.shape[-1]),
            d_model=int(cfg.patchtraj.d_model),
            n_heads=int(cfg.patchtraj.n_heads),
            n_layers=int(cfg.patchtraj.n_layers),
            dim_feedforward=int(cfg.patchtraj.dim_feedforward),
            dropout=float(cfg.patchtraj.dropout),
            patch_grid=patch_grid,
        ).to(device)
    else:
        model = PatchTrajPredictor(
            hidden_dim=int(tokens.shape[-1]),
            d_model=int(cfg.patchtraj.d_model),
            n_heads=int(cfg.patchtraj.n_heads),
            n_layers=int(cfg.patchtraj.n_layers),
            dim_feedforward=int(cfg.patchtraj.dim_feedforward),
            dropout=float(cfg.patchtraj.dropout),
            activation=str(cfg.patchtraj.activation),
        ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=int(cfg.training.epochs),
        steps_per_epoch=max(1, len(train_loader)),
        warmup_epochs=int(cfg.training.warmup_epochs),
    )

    wandb_run = maybe_init_wandb(cfg, output_dir)
    checkpoint_path = output_dir / "best_model.pt"
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    LOGGER.info(
        "Start training | windows=%d train_seq=%d val_seq=%d cache=%s",
        int(windows.shape[0]),
        len(train_ds),
        len(val_ds),
        cache_path,
    )

    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pi=pi,
            valid_mask=valid_mask,
            ot_soft_pi=ot_soft_pi,
            huber_delta=float(cfg.training.huber_delta),
            trimmed_ratio=float(cfg.training.trimmed_ratio),
        )
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            pi=pi,
            valid_mask=valid_mask,
            ot_soft_pi=ot_soft_pi,
            huber_delta=float(cfg.training.huber_delta),
            trimmed_ratio=float(cfg.training.trimmed_ratio),
        )

        lr_value = float(optimizer.param_groups[0]["lr"])
        LOGGER.info(
            "Epoch %d/%d | train=%.6f val=%.6f lr=%.6e",
            epoch,
            int(cfg.training.epochs),
            train_loss,
            val_loss,
            lr_value,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": lr_value,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "patch_grid": patch_grid,
                    "pi": pi,
                    "valid_mask": valid_mask,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                checkpoint_path,
            )
            LOGGER.info("Saved best checkpoint: %s", checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= int(cfg.training.patience):
                LOGGER.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch,
                    int(cfg.training.patience),
                )
                break

    LOGGER.info(
        "Training done | best_epoch=%d best_val=%.6f checkpoint=%s",
        best_epoch,
        best_val,
        checkpoint_path,
    )

    # --- Fit DualSignalScorer on training tokens ---
    dual_cfg = cfg.get("scoring", {}).get("dual_signal", {})
    if bool(dual_cfg.get("enabled", False)):
        dual_alpha = float(dual_cfg.get("alpha", 0.5))
        dual_scorer = DualSignalScorer(alpha=dual_alpha)
        dual_scorer.fit(tokens.numpy().astype(np.float64))

        # --- Compute validation scores for alpha selection in detect.py ---
        # Load best model for inference on validation split
        best_ckpt = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.eval()

        val_traj_chunks: list[np.ndarray] = []
        val_target_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for token_seq, target in val_loader:
                token_seq = token_seq.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                target = target.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                predicted = model(token_seq)
                batch_scores = compute_patchtraj_score(
                    predicted_tokens=predicted,
                    actual_tokens=target,
                    pi=pi,
                    valid_mask=valid_mask,
                )
                val_traj_chunks.append(
                    batch_scores.detach().cpu().numpy().astype(np.float64)
                )
                val_target_chunks.append(
                    target.detach().cpu().numpy().astype(np.float64)
                )

        val_traj_scores = np.concatenate(val_traj_chunks, axis=0)
        val_target_tokens = np.concatenate(val_target_chunks, axis=0)
        val_dist_scores = dual_scorer.score_distributional(val_target_tokens)

        # Freeze the (mu, sigma) used by fuse() so test-time fusion is leak-free
        # (uses train-validation reference statistics, not the test batch's own).
        dual_scorer.fit_normalizers(val_traj_scores, val_dist_scores)

        LOGGER.info(
            "Computed validation scores for alpha selection: "
            "traj=%d samples, dist=%d samples.",
            len(val_traj_scores),
            len(val_dist_scores),
        )

        # Append to existing checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec: trusted local checkpoint
        ckpt["dual_signal_state"] = dual_scorer.state_dict()
        ckpt["val_traj_scores"] = val_traj_scores
        ckpt["val_dist_scores"] = val_dist_scores
        torch.save(ckpt, checkpoint_path)
        LOGGER.info(
            "DualSignalScorer fitted and saved (alpha=%.2f) "
            "with validation scores for alpha selection.",
            dual_alpha,
        )

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
