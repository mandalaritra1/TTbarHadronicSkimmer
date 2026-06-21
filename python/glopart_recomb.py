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


# Engineered feature set for the data-space LOGISTIC fit (see fit_logistic). The
# 7 raw per-head log-scores PLUS two physically-motivated aggregate log-scores.
ENG_FEATURE_NAMES = [
    "log_TopSum", "log_TopbWqq", "log_TopbWq", "log_QCD",
    "log_Xqq", "log_Xcs", "log_Xbb", "log_Xcc", "log_XSum",
]
N_ENG = len(ENG_FEATURE_NAMES)  # 9


def build_features_eng(heads, eps=EPS):
    """Engineered (N, 9) log-score feature set used by the logistic fitter.

    The 7 raw per-head log-scores PLUS two aggregate log-scores: the
    hadronic-top sum ``log(TopbWqq + TopbWq)`` and the heavy-flavour-X sum
    ``log(Xqq + Xcs + Xbb + Xcc)``. The aggregates hand the *linear* logistic
    model the same "combined signal vs combined exotic" axes the baseline ratio
    uses, but WITHOUT forcing the baseline's fixed 1:1 TopbWqq:TopbWq mix or its
    QCD-only denominator — that freedom is where the boosted-tail gain comes from.

    Unlike the 7-D moment features (which feed the ntuple-free LDA and must keep
    a fixed length for the packed moment vector), this set is used only by the
    data-space logistic fit, so there is no moment-vector length constraint.
    """
    g = lambda name: np.asarray(_get_head(heads, name), dtype=np.float64)
    L = lambda name: np.log(g(name) + eps)
    top_sum = np.log(g("TopbWqq") + g("TopbWq") + eps)
    x_sum = np.log(g("Xqq") + g("Xcs") + g("Xbb") + g("Xcc") + eps)
    return np.stack([top_sum, L("TopbWqq"), L("TopbWq"), L("QCD"),
                     L("Xqq"), L("Xcs"), L("Xbb"), L("Xcc"), x_sum], axis=1)


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
# Logistic regression (data-space; the PRIMARY fitter)
# ---------------------------------------------------------------------------
# The LDA above is ntuple-free but assumes Gaussian per-class features, which the
# log-score heads only roughly satisfy (it underperforms the baseline at low pT).
# Logistic regression makes no Gaussian assumption and is consistently best; it
# needs the actual per-jet feature rows (an ntuple), not just moments. This is
# the verified winner for pT > 800 (+3-4% sig-eff at 0.5% mis-tag).
def fit_logistic(X, y, w=None, n_iter=80, l2=1e-3, balance=True):
    """Class-balanced, standardized logistic regression via IRLS (Newton steps).

    X : (N, F) feature matrix (use :func:`build_features_eng`).
    y : (N,) labels in {0 (bkg), 1 (sig)}.
    w : optional per-row weight (length N); defaults to 1.

    Returns a serialisable params dict ``{"mu","sd","beta"}`` such that the score
        s(Xn) = [1, (Xn - mu)/sd] @ beta
    is the log-odds (monotonic in probability) — all the ROC / fixed-mis-tag
    inversion needs. Features are standardized for conditioning; ``balance``
    reweights the two classes to equal total weight so the fit is prior-free (the
    absolute mis-tag WP is imposed downstream, not by the training prior). The
    ``l2`` ridge on the standardized coefficients stabilises collinear heads.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(X)) if w is None else np.asarray(w, dtype=np.float64).copy()
    if balance:
        for c in (0.0, 1.0):
            m = y == c
            s = w[m].sum()
            if s > 0:
                w[m] *= 0.5 / s
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = np.c_[np.ones(len(X)), (X - mu) / sd]
    beta = np.zeros(Xs.shape[1])
    for _ in range(n_iter):
        eta = np.clip(Xs @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        Wd = w * p * (1.0 - p)
        grad = Xs.T @ (w * (p - y)) + l2 * beta
        H = Xs.T @ (Xs * Wd[:, None]) + l2 * np.eye(Xs.shape[1])
        beta = beta - np.linalg.solve(H, grad)
    return {"mu": mu, "sd": sd, "beta": beta}


def apply_logistic(X, params):
    """Log-odds score for the logistic ``params`` from :func:`fit_logistic`."""
    X = np.asarray(X, dtype=np.float64)
    Xs = np.c_[np.ones(len(X)), (X - params["mu"]) / params["sd"]]
    return Xs @ params["beta"]


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


# --- exact (unbinned) ntuple-space metrics -------------------------------
# The functions above invert a *histogram* (the coffea-accumulated path). For the
# local per-jet study we have the raw score arrays, so we can invert exactly
# without binning artefacts. These are the counterparts used by the driver.
def weighted_quantile(x, q, w=None):
    """Weighted quantile of ``x`` at probability ``q`` in [0, 1] (interpolated)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    i = np.argsort(x)
    x, w = x[i], w[i]
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q * w.sum(), c, x))


def threshold_for_mistag(s_bkg, target_mistag, w_bkg=None):
    """Exact score threshold giving ``target_mistag`` tail efficiency on bkg."""
    return weighted_quantile(s_bkg, 1.0 - target_mistag, w_bkg)


def sigeff_at_mistag_unbinned(s_sig, s_bkg, target_mistag, w_sig=None, w_bkg=None):
    """Signal tail eff at the exact threshold for ``target_mistag`` on bkg.

    Returns (sig_eff, threshold).
    """
    thr = threshold_for_mistag(s_bkg, target_mistag, w_bkg)
    s_sig = np.asarray(s_sig, dtype=np.float64)
    w_sig = np.ones_like(s_sig) if w_sig is None else np.asarray(w_sig, dtype=np.float64)
    eff = float(w_sig[s_sig >= thr].sum() / w_sig.sum())
    return eff, thr


def roc_unbinned(s_sig, s_bkg, w_sig=None, w_bkg=None, n_points=400):
    """(mistag, sig_eff) ROC sampled at ``n_points`` thresholds (exact)."""
    s_sig = np.asarray(s_sig, dtype=np.float64)
    s_bkg = np.asarray(s_bkg, dtype=np.float64)
    w_sig = np.ones_like(s_sig) if w_sig is None else np.asarray(w_sig, dtype=np.float64)
    w_bkg = np.ones_like(s_bkg) if w_bkg is None else np.asarray(w_bkg, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_points)
    thr = np.unique(np.quantile(np.concatenate([s_sig, s_bkg]), qs))[::-1]
    tpr = np.array([w_sig[s_sig >= t].sum() for t in thr]) / w_sig.sum()
    fpr = np.array([w_bkg[s_bkg >= t].sum() for t in thr]) / w_bkg.sum()
    return fpr, tpr


def mistag_vs_value(value, score, threshold, edges, weight=None):
    """Tail mis-tag (fraction with score >= threshold) in bins of ``value``.

    The unbinned-score counterpart of :func:`mistag_vs_axis`, used to test
    mass-decorrelation directly from the per-jet arrays. Returns
    (centers, mistag, mistag_err) with a binomial error per bin.

    The error uses the *Kish effective sample size* per bin,
    ``n_eff = (Σw)^2 / Σw^2``, NOT Σw — essential once the jets carry real
    cross-section weights (Σw is inflated by the per-event weight and would make
    the error far too small, blowing up any flatness chi2). For unit weights
    n_eff reduces to the raw count.
    """
    value = np.asarray(value, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    weight = np.ones_like(value) if weight is None else np.asarray(weight, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tagged = score >= threshold
    mistag = np.full(len(centers), np.nan)
    err = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (value >= edges[i]) & (value < edges[i + 1])
        wm = weight[m]
        den = wm.sum()
        sumw2 = float((wm * wm).sum())
        if den > 0 and sumw2 > 0:
            p = weight[m & tagged].sum() / den
            mistag[i] = p
            n_eff = den * den / sumw2          # Kish effective sample size
            err[i] = np.sqrt(max(p * (1.0 - p), 0.0) / n_eff)
    return centers, mistag, err


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


def save_logistic_weights(path, pt_edges, params_per_bin,
                          feature_set="eng", metadata=None):
    """Persist per-pT logistic params (mu, sd, beta) from :func:`fit_logistic`."""
    data = {
        "feature_set": feature_set,
        "feature_names": list(ENG_FEATURE_NAMES) if feature_set == "eng" else None,
        "pt_edges": [float(e) for e in pt_edges],
        "eps": EPS,
        "params": {
            str(int(pb)): {
                "mu": np.asarray(p["mu"]).tolist(),
                "sd": np.asarray(p["sd"]).tolist(),
                "beta": np.asarray(p["beta"]).tolist(),
            }
            for pb, p in params_per_bin.items()
        },
        "metadata": metadata or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_logistic_weights(path):
    """Load logistic params saved by :func:`save_logistic_weights`.

    Returns a dict with ``params_per_bin`` mapping int pt-bin -> {mu, sd, beta}.
    """
    with open(path) as f:
        data = json.load(f)
    data["params_per_bin"] = {
        int(pb): {
            "mu": np.asarray(v["mu"], dtype=np.float64),
            "sd": np.asarray(v["sd"], dtype=np.float64),
            "beta": np.asarray(v["beta"], dtype=np.float64),
        }
        for pb, v in data["params"].items()
    }
    data["pt_edges"] = np.asarray(data["pt_edges"], dtype=np.float64)
    return data
