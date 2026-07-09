"""
Multi-domain feature extraction from Center-of-Pressure (COP) signals.

Implements a faithful, self-contained version of the three feature families
described in Paper 1 (Mengarelli et al., 2024), computed separately for the
antero-posterior (APc), medial-lateral (MLc) and statokinesigram (STKc)
components:

  1. Universal balance descriptors  (Yamamoto et al.):  SL, PS50, ANG, FLAT
  2. Stabilogram Diffusion Function (Collins & De Luca): CRT, MSD, DS, DL,
                                                          HS, HL, K, TL
  3. Structural / complexity measures:                   RQA (RR, DET, RT,
                                                          AVDL, MLL, ENT, TND,
                                                          LAM, MVL, TT), SAEN

For computational tractability the long 100 Hz recordings are decimated before
the O(N^2) recurrence and entropy steps; the embedding lag is rescaled to match
so the reconstructed dynamics are preserved.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# 1. Universal balance descriptors
# ---------------------------------------------------------------------------
def _power_spectrum_features(x, fs):
    """SL  = slope of the log-log power spectrum in the low-frequency band.
    PS50 = frequency below which 50% of the spectral power lies (median freq)."""
    x = x - np.mean(x)
    f, pxx = sp_signal.welch(x, fs=fs, nperseg=min(len(x), 1024))
    valid = f > 0
    f, pxx = f[valid], pxx[valid]
    # SL: linear fit of log(PSD) vs log(f) over the low band 0.15-0.6 Hz.
    band = (f >= 0.15) & (f <= 0.6)
    if band.sum() >= 3:
        sl = np.polyfit(np.log(f[band]), np.log(pxx[band] + 1e-20), 1)[0]
    else:
        sl = 0.0
    cumpow = np.cumsum(pxx) / (np.sum(pxx) + 1e-20)
    ps50 = float(f[np.searchsorted(cumpow, 0.5)]) if len(f) else 0.0
    return float(sl), ps50


def _ellipse_features(ap, ml):
    """From the 95% confidence ellipse of the statokinesigram:
    ANG  = orientation of the major axis w.r.t. the AP direction (deg)
    FLAT = flattening = 1 - minor/major axis length."""
    pts = np.vstack([ap - ap.mean(), ml - ml.mean()])
    cov = np.cov(pts)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major_vec = evecs[:, 0]
    ang = np.degrees(np.arctan2(abs(major_vec[1]), abs(major_vec[0])))
    minor, major = np.sqrt(max(evals[1], 1e-12)), np.sqrt(max(evals[0], 1e-12))
    flat = 1.0 - minor / major
    return float(ang), float(flat)


# ---------------------------------------------------------------------------
# 2. Stabilogram Diffusion Function (stochastic modelling of sway)
# ---------------------------------------------------------------------------
def _sdf_features(x, fs):
    """Return CRT, MSD, DS, DL, HS, HL, K, TL from the stabilogram diffusion
    function <Δx²>(Δt). Short- and long-term regions are separated at the
    transition point; a sigmoid fit on the log-log curve yields K and TL."""
    x = np.asarray(x, float)
    n = len(x)
    max_lag = int(min(n // 2, fs * 10))          # up to 10 s of lags
    lags = np.arange(1, max_lag)
    msd = np.empty(len(lags))
    for i, lag in enumerate(lags):
        d = x[lag:] - x[:-lag]
        msd[i] = np.mean(d * d)
    t = lags / fs

    # Transition point: where the local log-log slope drops the most.
    logt, logm = np.log(t + 1e-12), np.log(msd + 1e-12)
    dslope = np.gradient(logm, logt)
    trans = int(np.clip(np.argmin(np.gradient(dslope)) + 1, 3, len(t) - 3))
    crt = float(t[trans])
    msd_c = float(msd[trans])

    # Diffusion coefficients: slope of MSD vs Δt (linear) = 2·D per region.
    ds = float(np.polyfit(t[:trans], msd[:trans], 1)[0] / 2.0)
    dl = float(np.polyfit(t[trans:], msd[trans:], 1)[0] / 2.0)
    # Scaling (Hurst) exponents: half the log-log slope per region.
    hs = float(np.polyfit(logt[:trans], logm[:trans], 1)[0] / 2.0)
    hl = float(np.polyfit(logt[trans:], logm[trans:], 1)[0] / 2.0)

    # K = variance of displacement at Δt = 1 s (diffusion estimate).
    k = float(msd[np.argmin(np.abs(t - 1.0))])
    # TL = sigmoid mid-point of the log-log curve (persistent->antipersistent).
    y = (logm - logm.min()) / (np.ptp(logm) + 1e-12)
    tl = float(t[np.argmin(np.abs(y - 0.5))])
    return dict(CRT=crt, MSD=msd_c, DS=ds, DL=dl, HS=hs, HL=hl, K=k, TL=tl)


# ---------------------------------------------------------------------------
# 3. Structural / complexity measures
# ---------------------------------------------------------------------------
def _embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    if n <= 1:
        return np.empty((0, m))
    idx = np.arange(n)[:, None] + (np.arange(m) * tau)[None, :]
    return x[idx]


def _rqa_features(x, m=10, tau=4, rr_thresh=0.30, lmin=4):
    """Recurrence Quantification Analysis features from the recurrence plot.
    threshold epsilon = rr_thresh * mean pairwise distance (Paper 1)."""
    Y = _embed(np.asarray(x, float), m, tau)
    if len(Y) < lmin + 2:
        return {k: 0.0 for k in
                ["RR", "DET", "RT", "AVDL", "MLL", "ENT", "TND", "LAM", "MVL", "TT"]}
    # Pairwise Euclidean distance matrix.
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    eps = rr_thresh * d.mean()
    R = (d <= eps).astype(np.uint8)
    np.fill_diagonal(R, 0)                 # exclude line of identity
    N = R.shape[0]

    rr = R.sum() / (N * (N - 1))

    def _line_lengths(mat, diagonal):
        lengths = []
        rng = range(-N + 1, N) if diagonal else range(N)
        for off in rng:
            line = np.diag(mat, off) if diagonal else mat[:, off]
            c = 0
            for v in line:
                if v:
                    c += 1
                elif c:
                    lengths.append(c); c = 0
            if c:
                lengths.append(c)
        return np.array(lengths)

    diag = _line_lengths(R, True)
    diag = diag[diag >= lmin]
    vert = _line_lengths(R, False)
    vert = vert[vert >= lmin]

    total_pts = R.sum()
    det = diag.sum() / total_pts if total_pts else 0.0
    lam = vert.sum() / total_pts if total_pts else 0.0
    avdl = float(diag.mean()) if len(diag) else 0.0
    mll = float(diag.max()) if len(diag) else 0.0
    mvl = float(vert.max()) if len(vert) else 0.0
    tt = float(vert.mean()) if len(vert) else 0.0
    rt = det / rr if rr else 0.0

    # Shannon entropy of diagonal-line-length distribution.
    if len(diag):
        vals, counts = np.unique(diag, return_counts=True)
        p = counts / counts.sum()
        ent = float(-np.sum(p * np.log(p)))
    else:
        ent = 0.0

    # TND: trend of recurrence-point density vs distance from line of identity.
    offs = np.arange(1, N)
    dens = np.array([np.diag(R, o).mean() for o in offs])
    tnd = float(np.polyfit(offs, dens, 1)[0]) if len(offs) > 1 else 0.0

    return dict(RR=float(rr), DET=float(det), RT=float(rt), AVDL=avdl, MLL=mll,
                ENT=ent, TND=tnd, LAM=float(lam), MVL=mvl, TT=tt)


def _sample_entropy(x, m=2, r=0.3, tau=1):
    """Sample entropy: -log( A / B ), regularity statistic (Richman & Moorman).
    r is a fraction of the signal standard deviation."""
    x = np.asarray(x, float)
    tol = r * np.std(x)
    if tol == 0:
        return 0.0

    def _phi(mm):
        Y = _embed(x, mm, tau)
        if len(Y) < 2:
            return 0
        d = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)  # Chebyshev
        np.fill_diagonal(d, np.inf)
        return np.sum(d <= tol)

    B = _phi(m)
    A = _phi(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _decimate(x, factor):
    if factor <= 1:
        return x, 1
    return sp_signal.decimate(x, factor, ftype="fir", zero_phase=True), factor


def component_features(x, fs, ang=None, flat=None, decim=4):
    """All Paper-1 features for a single 1-D COP component."""
    feats = {}
    sl, ps50 = _power_spectrum_features(x, fs)
    feats["SL"], feats["PS50"] = sl, ps50
    if ang is not None:
        feats["ANG"], feats["FLAT"] = ang, flat
    feats.update(_sdf_features(x, fs))
    xd, factor = _decimate(x, decim)
    feats.update(_rqa_features(xd, m=10, tau=max(1, 15 // factor)))
    feats["SAEN"] = _sample_entropy(xd, m=2, r=0.3, tau=max(1, 15 // factor))
    return feats


def extract_cop_features(ap, ml, fs):
    """Feature dictionary for a subject, prefixed by COP component.

    Returns a flat {name: value} dict with keys like ``APc_SL`` / ``STKc_DET``.
    STKc uses the planar radial displacement series for the univariate
    measures, plus the shared ellipse geometry (ANG, FLAT).
    """
    ang, flat = _ellipse_features(ap, ml)
    stk = np.sqrt((ap - ap.mean()) ** 2 + (ml - ml.mean()) ** 2)  # planar displacement

    out = {}
    for name, comp, extra in (
        ("APc", ap, {}),
        ("MLc", ml, {}),
        ("STKc", stk, {"ang": ang, "flat": flat}),
    ):
        f = component_features(comp, fs, **extra)
        for k, v in f.items():
            out[f"{name}_{k}"] = v
    return out
