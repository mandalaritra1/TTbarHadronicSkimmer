"""GloParTv3 per-pT multiclass recombination — pure, ntuple-free library.

Companion to ``toptag_wp_processor.py``. The WP processor collapses GloParTv3
into the *single* hand-built ratio

    TopvsQCD = (TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD)

(see ``_topvsqcd_globalParT3`` / ``TTbarResProcessor._tscore``). This module asks
whether a *learned, per-pT linear recombination* of the raw GloParTv3 class heads
does better at fixed QCD mis-tag, while keeping the design ntuple-free.

Design (why this file is mostly NumPy):
---------------------------------------
The recombination weights are fit with a **Gaussian linear discriminant (LDA)**,
whose solution depends ONLY on per-class first and second moments
``(n, S=Σw·x, M=Σw·x·xᵀ)``. Those moments are additive, so they can be
accumulated in coffea like any histogram — no per-jet ntuples are needed. This
file provides:

  * the feature transform (raw head -> feature vector),
  * the baseline score (to beat),
  * moment packing/unpacking + the LDA solver,
  * per-pT fit / apply,
  * histogram-based metrics (fixed-mistag ROC inversion, mis-tag flatness),
  * portable JSON persistence of the fitted weights.

IMPORTANT data fact (verified on 2024 NanoAOD v15, not assumed):
    The mass-decorrelated ``FatJet_globalParT3_*`` heads are NOT a softmax. The
    17 class heads do not sum to 1 (mean ~0.82) and individual heads can exceed
    1 (e.g. Xqq up to ~1.4). They are non-negative "scores", and CMS documents
    them with "for sig vs bkg use sig/(sig+bkg)". Consequently the originally
    planned per-head logit ``log(P/(1-P))`` is undefined for P>1, so the DEFAULT
    feature transform here is the log-score ``log(P+eps)`` (robust for P>=0 and
    P>1). Log-ratios such as ``log(sig/bkg)`` are linear combinations of
    log-scores, hence already inside the LDA's solution span.
"""

import json

import numpy as np

# ---------------------------------------------------------------------------
# Head inventory
# ---------------------------------------------------------------------------
# The 7 raw heads used as model inputs, in a FIXED order. The order defines the
# layout of every feature vector, moment vector, and saved weight vector below.
# Two signal (hadronic-top) heads then five background-like heads.
RAW_HEADS = ["TopbWqq", "TopbWq", "QCD", "Xqq", "Xcs", "Xbb", "Xcc"]
N_FEAT = len(RAW_HEADS)  # 7

# The FULL set of GloParTv3 class heads (verified present in NanoAOD v15). Used
# ONLY for the non-negativity/finiteness sanity check — NOT as model inputs and
# NOT as a softmax (they do not sum to 1; see module docstring).
FULL_CLASS_HEADS = [
    "QCD",
    "TopbWev", "TopbWmv", "TopbWq", "TopbWqq", "TopbWtauhv",
    "XWW3q", "XWW4q", "XWWqqev", "XWWqqmv",
    "Xbb", "Xcc", "Xcs", "Xqq",
    "Xtauhtaue", "Xtauhtauh", "Xtauhtaum",
]

# Forbidden as inputs: built ratios and mass-correlated / regression outputs.
FORBIDDEN_HEADS = [
    "WvsQCD", "TopvsQCD",
    "withMassTopvsQCD", "withMassWvsQCD", "withMassZvsQCD",
    "massCorrGeneric", "massCorrX2p",
]

EPS = 1e-6

# Length of one packed moment vector: scalar n + length-7 S + 7x7 M.
MOMENT_VEC_LEN = 1 + N_FEAT + N_FEAT * N_FEAT  # 57


# ---------------------------------------------------------------------------
# Feature transforms
# ---------------------------------------------------------------------------
def _to_matrix(heads):
    """Stack a name->array mapping into an (N, 7) float64 matrix in RAW_HEADS order.

    ``heads`` may be a dict of numpy arrays or anything indexable by the head
    name returning a 1-D array (e.g. an awkward record already flattened to
    numpy). Branch names may be bare (``"Xqq"``) or prefixed
    (``"globalParT3_Xqq"`` / ``"FatJet_globalParT3_Xqq"``).
    """
    cols = []
    for h in RAW_HEADS:
        arr = _get_head(heads, h)
        cols.append(np.asarray(arr, dtype=np.float64))
    return np.stack(cols, axis=1)


def _get_head(heads, name):
    """Fetch one head allowing bare or prefixed branch names."""
    for key in (name, "globalParT3_" + name, "FatJet_globalParT3_" + name):
        try:
            if key in heads:
                return heads[key]
        except TypeError:
            pass
        try:
            return heads[key]
        except (KeyError, AttributeError, TypeError):
            continue
    raise KeyError(f"head {name!r} not found (tried bare and globalParT3_/FatJet_ prefixes)")


def feat_logscore(P, eps=EPS):
    """log(P + eps). Default. Defined for P>=0 and P>1; never NaN."""
    return np.log(np.asarray(P, dtype=np.float64) + eps)


def feat_clipped_logit(P, eps=EPS):
    """log(p/(1-p)) with p clipped to [eps, 1-eps].

    The originally-planned transform. Kept for comparison ONLY: it silently
    discards the genuine P>1 information (clips it to 1-eps), so it should not be
    the default for these non-softmax heads.
    """
    p = np.clip(np.asarray(P, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


# Per-head transforms that act elementwise on the (N, 7) matrix.
_ELEMENTWISE_TRANSFORMS = {
    "logscore": feat_logscore,
    "logit": feat_clipped_logit,
}

VALID_TRANSFORMS = ("logscore", "logratio", "logit")


def build_features(heads, transform="logscore", eps=EPS):
    """Return the (N, 7) feature matrix for the requested transform.

    transform:
      "logscore" (default) : log(P_i + eps)               — robust, accumulatable
      "logratio"           : log((P_i+eps)/(QCD+eps))     — CMS "sig/(sig+bkg)"
                             flavour; the QCD column is ~0 by construction (it
                             carries the removed overall scale) and is kept only
                             so the vector length stays 7; ridge handles it.
      "logit"              : clipped per-head logit (comparison only)
    """
    if transform not in VALID_TRANSFORMS:
        raise ValueError(f"transform must be one of {VALID_TRANSFORMS}, got {transform!r}")
    P = _to_matrix(heads)
    if transform == "logratio":
        qcd = np.asarray(_get_head(heads, "QCD"), dtype=np.float64)[:, None]
        return np.log((P + eps) / (qcd + eps))
    return _ELEMENTWISE_TRANSFORMS[transform](P, eps)


def baseline_topvsqcd(heads):
    """The score to beat: (TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD).

    Identical to ``_topvsqcd_globalParT3`` / ``TTbarResProcessor._tscore``.
    """
    top = np.asarray(_get_head(heads, "TopbWqq"), dtype=np.float64) + \
        np.asarray(_get_head(heads, "TopbWq"), dtype=np.float64)
    return top / (top + np.asarray(_get_head(heads, "QCD"), dtype=np.float64))


def assert_heads_sane(heads, atol_neg=-1e-4):
    """Sanity check for the raw heads: finite and (approximately) non-negative.

    Replaces the original (wrong) "sum of heads == 1" softmax assertion. These
    heads are NOT a softmax; the only invariants we can rely on are finiteness
    and non-negativity. Returns a dict of per-head diagnostics and raises on
    NaN/inf or clearly negative values.
    """
    diag = {}
    for h in RAW_HEADS:
        v = np.asarray(_get_head(heads, h), dtype=np.float64)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"head {h!r} contains non-finite values")
        if v.min() < atol_neg:
            raise ValueError(f"head {h!r} has negative values (min={v.min():.4g})")
        diag[h] = {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
    return diag


# ---------------------------------------------------------------------------
# Moments: pack / unpack / accumulate
# ---------------------------------------------------------------------------
# A packed moment vector stores, for one (jettype, pt-bin) class, the weighted
# sufficient statistics of the 7-D feature vector:
#     n = Σ w           (scalar)
#     S = Σ w·x         (length 7)
#     M = Σ w·x·xᵀ      (7x7, flattened)
# Packing them into ONE length-57 array means they merge across chunks by simple
# addition — exactly what coffea's accumulate does element-wise, so no custom
# accumulator class is needed.
def moment_vec(X, w=None):
    """Compute the packed length-57 moment vector for feature matrix X (N, 7).

    w : optional per-row weight (length N). Defaults to 1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_FEAT:
        raise ValueError(f"X must be (N, {N_FEAT}), got {X.shape}")
    if w is None:
        w = np.ones(len(X), dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)
    n = float(w.sum())
    S = (X * w[:, None]).sum(axis=0)                 # (7,)
    M = (X * w[:, None]).T @ X                        # (7,7) = Σ w x xᵀ
    out = np.empty(MOMENT_VEC_LEN, dtype=np.float64)
    out[0] = n
    out[1:1 + N_FEAT] = S
    out[1 + N_FEAT:] = M.ravel()
    return out


def unpack_moments(vec):
    """Inverse of :func:`moment_vec`: return (n, S(7), M(7,7))."""
    vec = np.asarray(vec, dtype=np.float64)
    n = float(vec[0])
    S = vec[1:1 + N_FEAT].copy()
    M = vec[1 + N_FEAT:].reshape(N_FEAT, N_FEAT).copy()
    return n, S, M


def empty_moment_vec():
    return np.zeros(MOMENT_VEC_LEN, dtype=np.float64)


# ---------------------------------------------------------------------------
# LDA fit from moments
# ---------------------------------------------------------------------------
def _cov_from_moments(n, S, M):
    mu = S / n
    return M / n - np.outer(mu, mu)


def moments_to_lda(vec_sig, vec_bkg, ridge=1e-4):
    """Fisher LDA from packed signal/background moment vectors.

    Returns (w, b) with the decision score s(x) = wᵀx + b, where the raw Fisher
    direction Σ⁻¹(μ_sig − μ_bkg) is rescaled to UNIT pooled projection variance
    (wᵀΣw = 1) and ``b = −½ (μ_sig + μ_bkg)ᵀ w`` puts the class midpoint at 0.

    Why standardize: scaling and bias are monotonic, so they change neither ROC
    nor the WP inversion. But a downstream ``sigmoid(s)`` filled into a fixed
    [0,1] histogram would saturate (all jets pile at ~1.0) if s had a large raw
    scale — destroying the tail resolution needed to invert a 0.5% mis-tag.
    Unit-variance standardization makes background ~N(−d/2, 1) and signal
    ~N(+d/2, 1) (d = Fisher separation), which sigmoid spreads cleanly across
    [0,1]. A ridge term ``ridge·I`` stabilises Σ (the 7 heads are linearly
    constrained; e.g. the logratio QCD feature is ~0).
    """
    n_s, S_s, M_s = unpack_moments(vec_sig)
    n_b, S_b, M_b = unpack_moments(vec_bkg)
    if n_s <= 0 or n_b <= 0:
        raise ValueError(f"empty class for LDA (n_sig={n_s}, n_bkg={n_b})")
    mu_s, mu_b = S_s / n_s, S_b / n_b
    Sig = (n_s * _cov_from_moments(n_s, S_s, M_s)
           + n_b * _cov_from_moments(n_b, S_b, M_b)) / (n_s + n_b)
    Sig = Sig + ridge * np.eye(N_FEAT)
    w = np.linalg.solve(Sig, mu_s - mu_b)
    proj_var = float(w @ Sig @ w)            # = Δμᵀ Σ⁻¹ Δμ = d² (squared separation)
    if proj_var > 1e-12:
        w = w / np.sqrt(proj_var)            # unit projection variance
    b = -0.5 * float(np.dot(mu_s + mu_b, w))
    return w, b


def moment_key(jettype, pt_bin):
    """Stable string key for the moment accumulator dict."""
    return f"{jettype}|{int(pt_bin)}"


def parse_moment_key(key):
    jettype, pt_bin = key.split("|")
    return jettype, int(pt_bin)


def fit_per_pt(moments, sig_jettype="matched", bkg_jettype="incl", ridge=1e-4):
    """Fit one (w, b) per pT bin from a moments dict.

    moments : {moment_key(jettype, pt_bin) -> packed vec}
    Returns : {pt_bin(int) -> (w(7,), b(float))} for every pT bin that has both
              a signal and a background class with non-zero counts.
    """
    pt_bins = sorted({parse_moment_key(k)[1] for k in moments})
    out = {}
    for pt in pt_bins:
        ks, kb = moment_key(sig_jettype, pt), moment_key(bkg_jettype, pt)
        if ks not in moments or kb not in moments:
            continue
        if moments[ks][0] <= 0 or moments[kb][0] <= 0:
            continue
        out[pt] = moments_to_lda(moments[ks], moments[kb], ridge=ridge)
    return out


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def pt_bin_index(pt_values, pt_edges):
    """Vectorised pT-bin index in [0, n_bins-1]; out-of-range clipped to ends."""
    idx = np.digitize(np.asarray(pt_values, dtype=np.float64), pt_edges) - 1
    return np.clip(idx, 0, len(pt_edges) - 2)


def apply_score(X, w, b=0.0, sigmoid=False):
    """s = X·w + b (optionally squashed). Monotonic sigmoid changes neither ROC
    nor the WP inversion; it only rescales the axis to [0, 1]."""
    s = np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + b
    if sigmoid:
        s = 1.0 / (1.0 + np.exp(-s))
    return s


def apply_per_pt(X, pt_values, pt_edges, weights_per_bin, sigmoid=False):
    """Score each jet with the weights of its own pT bin.

    weights_per_bin : {pt_bin -> (w, b)}. Jets whose pT bin has no fitted
    weights get NaN (caller decides how to handle; normally there are none after
    preselection).
    """
    X = np.asarray(X, dtype=np.float64)
    idx = pt_bin_index(pt_values, pt_edges)
    s = np.full(len(X), np.nan, dtype=np.float64)
    for pb, (w, b) in weights_per_bin.items():
        m = idx == pb
        if np.any(m):
            s[m] = apply_score(X[m], w, b, sigmoid=sigmoid)
    return s


# ---------------------------------------------------------------------------
# Histogram-based metrics
# ---------------------------------------------------------------------------
def efficiency_curve(counts, edges):
    """Tail efficiency vs threshold for "tag if disc >= threshold".

    Returns (thresholds, eff) where eff[i] = fraction of (weighted) entries with
    disc >= edges[i]. eff is monotonically non-increasing in the threshold.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return edges[:-1], np.zeros(len(counts))
    rev_cum = np.cumsum(counts[::-1])[::-1]           # entries with disc >= left edge
    return np.asarray(edges[:-1], dtype=np.float64), rev_cum / total


def threshold_for_target_eff(counts, edges, target_eff):
    """Threshold giving a target tail efficiency (e.g. 0.005 for 0.5% mis-tag)."""
    thr, eff = efficiency_curve(counts, edges)
    # eff is decreasing in thr; np.interp needs increasing x, so sort by eff.
    order = np.argsort(eff)
    return float(np.interp(target_eff, eff[order], thr[order]))


def eff_above_threshold(counts, edges, threshold):
    """Fraction of (weighted) entries with disc >= threshold."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    mask = np.asarray(edges[:-1], dtype=np.float64) >= threshold
    return float(counts[mask].sum() / total)


def roc_from_hists(sig_counts, bkg_counts, edges):
    """Return (mistag, sig_eff) arrays for a ROC, scanning the threshold."""
    _, sig_eff = efficiency_curve(sig_counts, edges)
    _, bkg_eff = efficiency_curve(bkg_counts, edges)
    return bkg_eff, sig_eff


def sigeff_at_fixed_mistag(sig_counts, bkg_counts, edges, target_mistag):
    """Signal efficiency at the threshold that yields ``target_mistag`` on bkg."""
    thr = threshold_for_target_eff(bkg_counts, edges, target_mistag)
    return eff_above_threshold(sig_counts, edges, thr), thr


def mistag_vs_axis(counts2d, disc_edges, threshold):
    """Per-slice mis-tag from a (n_axis, n_disc) count matrix at a fixed threshold.

    counts2d[i, :] is the disc distribution in axis-slice i (e.g. an m_tt or m_SD
    slice). Returns the fraction above ``threshold`` in each slice.
    """
    counts2d = np.asarray(counts2d, dtype=np.float64)
    above = np.asarray(disc_edges[:-1], dtype=np.float64) >= threshold
    num = counts2d[:, above].sum(axis=1)
    den = counts2d.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def flatness(x, y, yerr=None):
    """Quantify how flat y(x) is. Returns dict with constant-fit chi2/ndf and a
    straight-line slope (both small => flat => more factorizable background)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)[good]
        w = 1.0 / np.where(yerr > 0, yerr ** 2, np.inf)
    else:
        w = np.ones_like(y)
    if len(y) < 2:
        return {"chi2_const": np.nan, "ndf": 0, "slope": np.nan, "mean": float(y.mean()) if len(y) else np.nan}
    mean = float(np.sum(w * y) / np.sum(w))
    chi2_const = float(np.sum(w * (y - mean) ** 2))
    ndf = len(y) - 1
    # weighted linear slope
    xm = np.sum(w * x) / np.sum(w)
    ym = mean
    sxx = np.sum(w * (x - xm) ** 2)
    sxy = np.sum(w * (x - xm) * (y - ym))
    slope = float(sxy / sxx) if sxx > 0 else np.nan
    return {"chi2_const": chi2_const, "ndf": ndf, "chi2_per_ndf": chi2_const / ndf,
            "slope": slope, "mean": mean}


# ---------------------------------------------------------------------------
# Persistence (portable JSON; no pickle)
# ---------------------------------------------------------------------------
def save_weights(path, pt_edges, weights_per_bin, transform="logscore",
                 bias_convention="unit_var_equal_prior_midpoint", metadata=None):
    """Persist fitted per-pT weights to JSON."""
    data = {
        "raw_heads": list(RAW_HEADS),
        "pt_edges": [float(e) for e in pt_edges],
        "transform": transform,
        "eps": EPS,
        "bias_convention": bias_convention,
        "weights": {
            str(int(pb)): {"w": np.asarray(w).tolist(), "b": float(b)}
            for pb, (w, b) in weights_per_bin.items()
        },
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_weights(path):
    """Load weights saved by :func:`save_weights`.

    Returns a dict with ``weights_per_bin`` mapping int pt-bin -> (w(7,), b).
    """
    with open(path) as f:
        data = json.load(f)
    data["weights_per_bin"] = {
        int(pb): (np.asarray(v["w"], dtype=np.float64), float(v["b"]))
        for pb, v in data["weights"].items()
    }
    data["pt_edges"] = np.asarray(data["pt_edges"], dtype=np.float64)
    return data
