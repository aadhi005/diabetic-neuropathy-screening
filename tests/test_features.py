"""
Feature-extraction tests.

These assert against signals whose properties are known analytically, because
a subtly wrong feature still returns a plausible number -- the model will
happily learn from it and report a confident, meaningless accuracy.
"""

import numpy as np
import pytest

from neuroscreen.features import (_ellipse_features, _power_spectrum_features,
                                  _sample_entropy, extract_cop_features)
from neuroscreen.signals import FS


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --------------------------------------------------------------------------
# Spectral features -- _power_spectrum_features returns (SL, PS50)
# --------------------------------------------------------------------------
def test_median_frequency_higher_for_faster_signal():
    """A 4 Hz oscillation must have a higher median frequency than a 0.5 Hz one."""
    t = np.arange(0, 60, 1 / FS)
    _, ps50_slow = _power_spectrum_features(np.sin(2 * np.pi * 0.5 * t), FS)
    _, ps50_fast = _power_spectrum_features(np.sin(2 * np.pi * 4.0 * t), FS)
    assert ps50_fast > ps50_slow


def test_spectral_slope_flatter_for_white_noise_than_brown(rng):
    """White noise has a flat spectrum; integrated (brown) noise falls off."""
    white = rng.normal(size=12000)
    brown = np.cumsum(rng.normal(size=12000))
    sl_white, _ = _power_spectrum_features(white, FS)
    sl_brown, _ = _power_spectrum_features(brown, FS)
    assert sl_white > sl_brown, f"white {sl_white:.2f} should exceed brown {sl_brown:.2f}"


def test_spectral_features_finite_on_white_noise(rng):
    sl, ps50 = _power_spectrum_features(rng.normal(size=6000), FS)
    assert np.isfinite(sl) and np.isfinite(ps50)


# --------------------------------------------------------------------------
# Sway geometry -- _ellipse_features returns (ANG, FLAT)
# --------------------------------------------------------------------------
def test_flatness_near_zero_for_isotropic_sway(rng):
    """Equal spread on both axes => circular ellipse => flattening ~0."""
    n = 20000
    _, flat = _ellipse_features(rng.normal(size=n), rng.normal(size=n))
    assert flat < 0.15, f"expected near-circular, got flattening {flat:.3f}"


def test_flatness_near_one_for_collapsed_sway(rng):
    """Sway confined to one axis => degenerate ellipse => flattening ~1."""
    n = 20000
    _, flat = _ellipse_features(rng.normal(size=n), 0.02 * rng.normal(size=n))
    assert flat > 0.9, f"expected near-degenerate, got flattening {flat:.3f}"


def test_ellipse_flatness_is_scale_invariant(rng):
    """Flattening is a shape ratio: scaling both axes must not change it."""
    n = 8000
    ap, ml = rng.normal(size=n), 0.4 * rng.normal(size=n)
    _, a = _ellipse_features(ap, ml)
    _, b = _ellipse_features(5 * ap, 5 * ml)
    assert a == pytest.approx(b, abs=1e-9)


def test_ellipse_angle_tracks_dominant_axis(rng):
    """Sway mostly along ML should orient differently from sway along AP."""
    n = 8000
    ang_ap, _ = _ellipse_features(rng.normal(size=n), 0.05 * rng.normal(size=n))
    ang_ml, _ = _ellipse_features(0.05 * rng.normal(size=n), rng.normal(size=n))
    assert abs(ang_ap - ang_ml) > 45


# --------------------------------------------------------------------------
# Complexity
# --------------------------------------------------------------------------
def test_sample_entropy_lower_for_periodic_than_noise(rng):
    """A pure sine is perfectly predictable; white noise is not."""
    t = np.arange(0, 40, 1 / FS)
    periodic = _sample_entropy(np.sin(2 * np.pi * 1.0 * t))
    noise = _sample_entropy(rng.normal(size=len(t)))
    assert periodic < noise, f"sine {periodic:.3f} should be < noise {noise:.3f}"


# --------------------------------------------------------------------------
# Full extraction contract
# --------------------------------------------------------------------------
def test_extract_returns_finite_named_features(rng):
    feats = extract_cop_features(rng.normal(size=6000), rng.normal(size=6000), FS)
    assert isinstance(feats, dict) and feats
    bad = {k: v for k, v in feats.items() if not np.isfinite(v)}
    assert not bad, f"non-finite features: {list(bad)[:5]}"


def test_extract_covers_all_three_components(rng):
    """Paper 1 computes its families on APc, MLc and the statokinesigram."""
    feats = extract_cop_features(rng.normal(size=6000), rng.normal(size=6000), FS)
    for prefix in ("APc_", "MLc_", "STKc_"):
        assert any(k.startswith(prefix) for k in feats), f"no {prefix} features"


def test_extract_includes_each_paper_feature_family(rng):
    """Universal descriptors, stabilogram diffusion and RQA must all appear."""
    feats = extract_cop_features(rng.normal(size=6000), rng.normal(size=6000), FS)
    for key in ("STKc_SL", "STKc_PS50", "STKc_ANG", "STKc_FLAT",   # universal
                "STKc_CRT", "STKc_MSD", "STKc_DS", "STKc_HL",      # diffusion
                "STKc_RR", "STKc_DET", "STKc_LAM", "STKc_SAEN"):   # structural
        assert key in feats, f"missing {key}"


def test_extract_is_deterministic(rng):
    ap, ml = rng.normal(size=4000), rng.normal(size=4000)
    a = extract_cop_features(ap, ml, FS)
    b = extract_cop_features(ap.copy(), ml.copy(), FS)
    assert a == pytest.approx(b)


def test_mean_square_displacement_grows_with_sway(rng):
    """MSD is the magnitude descriptor: bigger sway must give a bigger value."""
    ap, ml = rng.normal(size=6000), rng.normal(size=6000)
    calm = extract_cop_features(ap, ml, FS)
    unsteady = extract_cop_features(3 * ap, 3 * ml, FS)
    assert unsteady["STKc_MSD"] > calm["STKc_MSD"]
