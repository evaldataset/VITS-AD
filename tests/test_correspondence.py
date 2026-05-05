"""Tests for src/rendering/token_correspondence.py — π map correctness."""

from __future__ import annotations

import numpy as np
import pytest

from src.rendering.token_correspondence import (
    compute_correspondence_map,
    get_valid_patch_count,
)


# ---------------------------------------------------------------------------
# line_plot correspondence
# ---------------------------------------------------------------------------


def test_line_plot_pi_shape() -> None:
    pi, valid_mask = compute_correspondence_map("line_plot", window_size=100, stride=10)
    assert pi.shape == (196,)
    assert valid_mask.shape == (196,)


def test_line_plot_pi_values() -> None:
    """window=100, stride=10, grid=(14,14):
    delta_col = round(14 * 10 / 100) = round(1.4) = 1
    Patches in column 0 → new_col = -1 → invalid
    Patches in columns 1-13 → valid
    Valid count = 14 * 13 = 182
    """
    pi, valid_mask = compute_correspondence_map("line_plot", window_size=100, stride=10)
    valid_count = get_valid_patch_count(valid_mask)
    assert valid_count == 182

    # Column 0 patches should be invalid
    for row in range(14):
        idx = row * 14 + 0  # column 0
        assert not valid_mask[idx]
        assert pi[idx] == -1

    # Column 5 should map to column 4 (shift by 1)
    for row in range(14):
        src_idx = row * 14 + 5
        expected_dst = row * 14 + 4
        assert valid_mask[src_idx]
        assert pi[src_idx] == expected_dst


def test_line_plot_pi_identity_no_shift() -> None:
    """stride=1, window=1000 → delta_col = round(14/1000) = 0 → identity map."""
    pi, valid_mask = compute_correspondence_map("line_plot", window_size=1000, stride=1)
    assert np.all(valid_mask)
    expected = np.arange(196)
    np.testing.assert_array_equal(pi, expected)


def test_line_plot_pi_all_invalid() -> None:
    """stride=99, window=100 → delta_col = round(14*99/100) = round(13.86) = 14
    All columns shift by 14 → all invalid."""
    pi, valid_mask = compute_correspondence_map("line_plot", window_size=100, stride=99)
    assert get_valid_patch_count(valid_mask) == 0
    assert np.all(pi == -1)


# ---------------------------------------------------------------------------
# GAF correspondence
# ---------------------------------------------------------------------------


def test_gaf_pi_shape() -> None:
    pi, valid_mask = compute_correspondence_map("gaf", window_size=100, stride=10)
    assert pi.shape == (196,)
    assert valid_mask.shape == (196,)


def test_gaf_pi_diagonal_shift() -> None:
    """window=100, stride=10 → delta = round(14*10/100) = 1
    Row 0 and col 0 become invalid (new_row or new_col = -1)."""
    pi, valid_mask = compute_correspondence_map("gaf", window_size=100, stride=10)

    # Row 0 patches: all invalid (new_row = -1)
    for col in range(14):
        assert not valid_mask[0 * 14 + col]

    # Column 0 patches: all invalid (new_col = -1)
    for row in range(14):
        assert not valid_mask[row * 14 + 0]

    # Patch (2, 3) → (1, 2)
    src = 2 * 14 + 3
    dst = 1 * 14 + 2
    assert valid_mask[src]
    assert pi[src] == dst


def test_gaf_pi_valid_count() -> None:
    """For delta=1 on a 14x14 grid: valid = (14-1)^2 = 169."""
    pi, valid_mask = compute_correspondence_map("gaf", window_size=100, stride=10)
    assert get_valid_patch_count(valid_mask) == 169


def test_recurrence_plot_same_as_gaf() -> None:
    """recurrence_plot uses same diagonal shift as GAF."""
    pi_gaf, mask_gaf = compute_correspondence_map("gaf", window_size=100, stride=10)
    pi_rp, mask_rp = compute_correspondence_map(
        "recurrence_plot", window_size=100, stride=10
    )
    np.testing.assert_array_equal(pi_gaf, pi_rp)
    np.testing.assert_array_equal(mask_gaf, mask_rp)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_renderer() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        compute_correspondence_map("spectrogram", window_size=100, stride=10)


def test_stride_ge_window() -> None:
    with pytest.raises(ValueError, match="stride"):
        compute_correspondence_map("line_plot", window_size=100, stride=100)
    with pytest.raises(ValueError, match="stride"):
        compute_correspondence_map("line_plot", window_size=100, stride=150)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def test_get_valid_patch_count() -> None:
    mask = np.array([True, False, True, True, False])
    assert get_valid_patch_count(mask) == 3


def test_get_valid_patch_count_all_true() -> None:
    mask = np.ones(196, dtype=bool)
    assert get_valid_patch_count(mask) == 196


def test_get_valid_patch_count_all_false() -> None:
    mask = np.zeros(196, dtype=bool)
    assert get_valid_patch_count(mask) == 0
