"""Multi-view rendering: combine multiple renderers into a single image."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

FloatImage = NDArray[np.float32]

# Registry of available renderers — populated lazily to avoid circular imports
_RENDERER_REGISTRY: dict[str, Callable[..., FloatImage]] = {}


def _get_renderer(name: str) -> Callable[..., FloatImage]:
    """Get renderer function by name, importing lazily.

    Args:
        name: Renderer name ('line_plot', 'gaf', 'recurrence_plot').

    Returns:
        Renderer function with signature (window: ndarray, **kwargs) -> FloatImage.

    Raises:
        ValueError: If renderer name is unknown.
    """
    if name in _RENDERER_REGISTRY:
        return _RENDERER_REGISTRY[name]

    if name == "line_plot":
        from src.rendering.line_plot import render_line_plot

        _RENDERER_REGISTRY[name] = render_line_plot
    elif name == "line_plot_fast":
        from src.rendering.line_plot_fast import render_line_plot_fast

        _RENDERER_REGISTRY[name] = render_line_plot_fast
    elif name == "gaf":
        from src.rendering.gaf import render_gaf

        _RENDERER_REGISTRY[name] = render_gaf
    elif name == "recurrence_plot":
        from src.rendering.recurrence_plot import render_recurrence_plot

        _RENDERER_REGISTRY[name] = render_recurrence_plot
    else:
        raise ValueError(
            f"Unknown renderer '{name}'. "
            "Supported: ['line_plot', 'line_plot_fast', 'gaf', 'recurrence_plot']."
        )

    return _RENDERER_REGISTRY[name]


def render_multi_view(
    window: np.ndarray,
    renderers: list[str] | None = None,
    image_size: int = 224,
    renderer_kwargs: dict[str, dict[str, Any]] | None = None,
) -> FloatImage:
    """Render a window using multiple renderers and stack channel-wise.

    Each renderer produces a (3, H, W) image. The outputs are concatenated
    along the channel axis to form a (3*V, H, W) tensor where V is the
    number of views.

    Args:
        window: Time series window of shape (L, D).
        renderers: List of renderer names. Defaults to all three.
        image_size: Output spatial size per view.
        renderer_kwargs: Per-renderer keyword arguments.
            Keys are renderer names, values are kwarg dicts.

    Returns:
        Multi-view image of shape (3*V, image_size, image_size), float32.

    Raises:
        ValueError: If inputs are invalid.
    """
    if renderers is None:
        renderers = ["line_plot", "gaf", "recurrence_plot"]
    if len(renderers) == 0:
        raise ValueError("renderers list must be non-empty.")
    if renderer_kwargs is None:
        renderer_kwargs = {}

    window_array = np.asarray(window)
    if window_array.ndim != 2:
        raise ValueError(
            f"window must have shape (L, D), got ndim={window_array.ndim}."
        )

    views: list[FloatImage] = []
    for renderer_name in renderers:
        render_fn = _get_renderer(renderer_name)
        kwargs = {"image_size": image_size}
        kwargs.update(renderer_kwargs.get(renderer_name, {}))
        view = render_fn(window=window_array, **kwargs)
        views.append(view)

    return np.concatenate(views, axis=0).astype(np.float32, copy=False)


def render_multi_view_batch(
    windows: np.ndarray,
    renderers: list[str] | None = None,
    image_size: int = 224,
    renderer_kwargs: dict[str, dict[str, Any]] | None = None,
) -> FloatImage:
    """Render batch of windows with multiple renderers.

    Args:
        windows: Batch of windows, shape (N, L, D).
        renderers: List of renderer names.
        image_size: Output spatial size per view.
        renderer_kwargs: Per-renderer keyword arguments.

    Returns:
        Multi-view images of shape (N, 3*V, image_size, image_size), float32.

    Raises:
        ValueError: If inputs are invalid.
    """
    windows_array = np.asarray(windows)
    if windows_array.ndim != 3:
        raise ValueError(
            f"windows must have shape (N, L, D), got ndim={windows_array.ndim}."
        )
    if windows_array.shape[0] == 0:
        raise ValueError("windows batch is empty (N=0).")

    rendered = [
        render_multi_view(
            window=w,
            renderers=renderers,
            image_size=image_size,
            renderer_kwargs=renderer_kwargs,
        )
        for w in windows_array
    ]
    return np.stack(rendered, axis=0).astype(np.float32, copy=False)


def compute_view_disagreement(
    per_view_scores: np.ndarray,
) -> np.ndarray:
    """Compute view disagreement as standard deviation across views.

    Higher disagreement between views indicates higher anomaly likelihood.
    This is the ViewDisagree signal from RESEARCH.md.

    Args:
        per_view_scores: Anomaly scores per view, shape (V, T).
            V = number of views, T = number of time steps.

    Returns:
        Disagreement scores of shape (T,).

    Raises:
        ValueError: If input is invalid.
    """
    if not isinstance(per_view_scores, np.ndarray):
        raise ValueError(
            f"per_view_scores must be numpy.ndarray, got {type(per_view_scores)!r}."
        )
    if per_view_scores.ndim != 2:
        raise ValueError(
            f"per_view_scores must have shape (V, T), got ndim={per_view_scores.ndim}."
        )
    if per_view_scores.shape[0] < 2:
        raise ValueError("Need at least 2 views for disagreement computation.")

    return np.std(per_view_scores, axis=0).astype(np.float64)
