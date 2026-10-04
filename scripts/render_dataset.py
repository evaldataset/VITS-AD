"""Pre-render time series windows as images for a dataset.

Usage:
    python scripts/render_dataset.py data=smd render=line_plot

This script:
1. Loads dataset (train + test windows)
2. Renders each window as an image using the configured renderer
3. Saves rendered images as .npy files for fast loading during training

Output structure:
    data/processed/{dataset}/
    ├── images_train_{render_name}_{entity}.npy   # (N_train, 3, 224, 224) float32
    └── images_test_{render_name}_{entity}.npy    # (N_test, 3, 224, 224) float32
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.smd import SMDDataset
from src.rendering.line_plot import render_line_plot_batch
from src.utils.reproducibility import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Batch size for rendering (to manage memory)
_RENDER_BATCH_SIZE = 128


def _load_dataset(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Load dataset based on config and return train/test windows + entity name.

    Args:
        cfg: Hydra config with data section.

    Returns:
        Tuple of (train_windows, test_windows, entity_name).

    Raises:
        ValueError: If dataset name is not supported.
    """
    dataset_name = cfg.data.name

    if dataset_name == "smd":
        entity = cfg.data.get("entity", "machine-1-1")
        dataset = SMDDataset(
            raw_dir=cfg.data.raw_dir,
            entity=entity,
            window_size=cfg.data.window_size,
            stride=cfg.data.stride,
            normalize=cfg.data.get("normalize", True),
            norm_method=cfg.data.get("norm_method", "standard"),
        )
        return dataset.train_windows, dataset.test_windows, entity

    raise ValueError(f"Unsupported dataset: '{dataset_name}'. Supported: ['smd']")


def _render_windows_batched(
    windows: np.ndarray,
    cfg: DictConfig,
    desc: str = "Rendering",
) -> np.ndarray:
    """Render windows in batches with progress bar.

    Args:
        windows: Windows of shape (N, L, D).
        cfg: Hydra config with render section.
        desc: Progress bar description.

    Returns:
        Rendered images of shape (N, 3, image_size, image_size), float32.
    """
    renderer_name = cfg.render.name
    num_windows = windows.shape[0]

    if renderer_name == "line_plot":
        render_kwargs = {
            "image_size": cfg.render.image_size,
            "dpi": cfg.render.get("dpi", 100),
            "colormap": cfg.render.get("colormap", "tab10"),
            "line_width": cfg.render.get("line_width", 1.0),
            "background_color": cfg.render.get("background_color", "white"),
            "show_axes": cfg.render.get("show_axes", False),
            "show_grid": cfg.render.get("show_grid", False),
        }
    else:
        raise ValueError(
            f"Unsupported renderer: '{renderer_name}'. Supported: ['line_plot']"
        )

    all_images: list[np.ndarray] = []
    batch_size = _RENDER_BATCH_SIZE

    for start_idx in tqdm(
        range(0, num_windows, batch_size),
        desc=desc,
        unit="batch",
    ):
        end_idx = min(start_idx + batch_size, num_windows)
        batch_windows = windows[start_idx:end_idx]

        if renderer_name == "line_plot":
            batch_images = render_line_plot_batch(
                windows=batch_windows, **render_kwargs
            )
        else:
            raise ValueError(f"Unsupported renderer: '{renderer_name}'")

        all_images.append(batch_images)

    return np.concatenate(all_images, axis=0).astype(np.float32, copy=False)


def _save_images(
    images: np.ndarray,
    output_path: Path,
) -> None:
    """Save rendered images to disk.

    Args:
        images: Image array to save.
        output_path: Destination .npy file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), images)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Saved %d images (%s) to %s (%.1f MB)",
        images.shape[0],
        "x".join(str(d) for d in images.shape),
        output_path,
        size_mb,
    )


@hydra.main(
    config_path="../configs",
    config_name="experiment/patchtraj_default",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Pre-render dataset images.

    Args:
        cfg: Hydra configuration.
    """
    seed_everything(cfg.training.seed)
    logger.info(
        "Pre-rendering dataset: %s with renderer: %s", cfg.data.name, cfg.render.name
    )

    train_windows, test_windows, entity = _load_dataset(cfg)
    logger.info(
        "Loaded %d train windows and %d test windows (entity=%s)",
        train_windows.shape[0],
        test_windows.shape[0],
        entity,
    )

    processed_dir = Path(cfg.data.processed_dir)
    render_name = cfg.render.name

    # Check if already rendered
    train_path = processed_dir / f"images_train_{render_name}_{entity}.npy"
    test_path = processed_dir / f"images_test_{render_name}_{entity}.npy"

    if train_path.exists() and test_path.exists():
        logger.info("Rendered images already exist. Skipping. Delete to re-render.")
        logger.info("  Train: %s", train_path)
        logger.info("  Test:  %s", test_path)
        return

    # Render train windows
    logger.info("Rendering %d training windows...", train_windows.shape[0])
    train_images = _render_windows_batched(
        windows=train_windows, cfg=cfg, desc="Render train"
    )
    _save_images(train_images, train_path)
    del train_images  # Free memory

    # Render test windows
    logger.info("Rendering %d test windows...", test_windows.shape[0])
    test_images = _render_windows_batched(
        windows=test_windows, cfg=cfg, desc="Render test"
    )
    _save_images(test_images, test_path)

    logger.info("Pre-rendering complete for %s/%s.", cfg.data.name, entity)


if __name__ == "__main__":
    main()
