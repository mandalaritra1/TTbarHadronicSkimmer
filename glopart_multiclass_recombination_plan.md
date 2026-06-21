# Plan: Per-$p_T$ GloParTv3 Multiclass Recombination Study (2024)

**Status: plan only — no analysis code changed yet. This document is written so it
can be handed to Claude Code to build a self-contained *test/study* (not a
production change to the analysis).**

Companion to `toptag_wp_derivation_plan.md`. That plan derives working points for
the *single* `TopvsQCD` score; this plan asks whether a *learned per-$p_T$
recombination of the raw GloParTv3 class heads* does better, while staying
compatible with the data-driven (2DAlphabet) background.

**Primary design choice: this study is done entirely in coffea with histograms +
small moment accumulators — no per-jet ntuples.** The recombination weights are
fit from accumulated per-class moments via a linear discriminant (LDA), not from
iterative logistic regression over per-jet rows. Ntuples appear only as an
*optional* upgrade (§4.3) on a small skim.

---

## 1. Problem statement

In `TTbarResProcessor._tscore` (`ttbarprocessor.py:274`) and in
`_topvsqcd_globalParT3` (`python/toptag_wp_processor.py:54`) we collapse GloParTv3
into a single hand-built ratio:

$$
D_{\mathrm{top}} = \frac{P_{TopbWqq} + P_{TopbWq}}
                        {P_{TopbWqq} + P_{TopbWq} + P_{QCD}}.
$$

Two choices are hard-coded into that ratio:

1. **Equal (1:1) weighting** of the fully-merged 3-prong head `TopbWqq` and the
   partially-merged head `TopbWq`.
2. **A denominator that normalizes against `QCD` only**, ignoring the other
   background-like heads the network produces (`Xqq`, `Xcs`, `Xbb`, `Xcc` — the
   two-prong family).

The GloParTv3 heads are network classification scores. Note (§3) that the
**mass-decorrelated** heads we use are *unnormalized* — non-negative but **not** a
probability simplex (they need not sum to 1, and some, e.g. `Xqq`, can exceed 1).
Even setting normalization aside, they are **not calibrated for our phase space**:
GloParT's outputs are conditioned on its training mixture, not on our
ultra-high-$p_T$, $H_T>1500$ GeV regime. Two physics
effects make the fixed 1:1 ratio sub-optimal in our tail:

- As $p_T$ grows the top collapses and a prong leaks out of the AK8 cone, so real
  tops migrate `TopbWqq` → `TopbWq`. But **so do QCD fakes** (a genuine 2-prong
  QCD jet also lights up `TopbWq`, `Xqq`). The optimal mix of `TopbWqq`/`TopbWq`
  is therefore **$p_T$-dependent** and not 1:1.
- The two-prong heads (`Xqq`, `Xcs`, `Xbb`, `Xcc`) carry "this is a 2-prong
  dijet, not a 3-prong top" information that the current ratio throws away.

**Hypothesis:** a tiny, per-$p_T$ **linear** recombination of the *raw* heads
(freezing the transformer; only ~7 weights per $p_T$ bin are fit) recovers a few
percent background rejection at fixed signal efficiency, and — more importantly —
may flatten the QCD mis-tag rate versus $m_{t\bar t}$ and $m_{\mathrm{SD}}$,
allowing the 2DAlphabet transfer-function order to drop.

---

## 2. Goal and decision rule

Build a reproducible **histogram-only** study that, for 2024:

1. accumulates per-class, per-$p_T$ **moments** of the head log-scores (sufficient
   statistics for an LDA fit) and the baseline score histograms — no ntuples;
2. fits a per-$p_T$ linear recombination `s(jet)` of the raw heads (top vs QCD)
   from those moments, offline;
3. fills score histograms for `s` in a second pass and compares against the
   baseline `TopvsQCD` **at fixed QCD mis-tag efficiency** (the convention already
   used in `toptag_wp_derivation_plan.md`), reporting signal-efficiency gain per
   $p_T$ bin (ROC);
4. measures the **deciding metric**: QCD mis-tag flatness vs $m_{t\bar t}$ and
   $m_{\mathrm{SD}}$ for `s` vs baseline, from `score_vs_mtt` / `score_vs_msd`
   histograms.

**Decision rule (what success means):**

| Outcome | Interpretation | Action |
|---|---|---|
| `s` flattens mis-tag vs $m_{t\bar t}$/$m_{\mathrm{SD}}$ **and** (optional Stage 6) lowers the 2DAlphabet F-test order | Real systematic reduction | Worth adopting; accept the calibration/SF cost (see §11) |
| `s` only improves ROC, no flattening | Marginal | Keep central WP for the result; fold learning into the Tier-1 event classifier |
| `s` worse or re-correlates mass | Hypothesis rejected | Document and stop |

ROC gain is the warm-up; **mis-tag flatness is the go/no-go.**

---

## 3. GloParTv3 head inventory — what to use and avoid

From the 2024 NanoAOD v15 `FatJet_globalParT3_*` documentation:

**Use as model inputs (raw, mass-decorrelated class heads):**

| Side | Heads |
|---|---|
| Signal (hadronic top) | `TopbWqq`, `TopbWq` |
| Background | `QCD`, `Xqq`, `Xcs`, `Xbb`, `Xcc` |

**The full GloParTv3 class set (17 heads).** For reference, the complete label
set is: `QCD`; five top modes `TopbWqq`, `TopbWq`, `TopbWev`, `TopbWmv`,
`TopbWtauhv`; four `XWW*` modes `XWW3q`, `XWW4q`, `XWWqqev`, `XWWqqmv`; four
2-prong `Xbb`, `Xcc`, `Xcs`, `Xqq`; three `Xtau*` modes `Xtauhtaue`, `Xtauhtauh`,
`Xtauhtaum`. Only the 7 in the table above are used as model inputs.

**Do NOT use as inputs:**

- `TopbWev`, `TopbWmv`, `TopbWtauhv` (semileptonic tops), `XWW*`, `Xtau*` —
  irrelevant to the all-hadronic top-vs-QCD problem (optional: monitor only).
- `WvsQCD`, and our own `TopvsQCD` — these are **already-built ratios**
  (`WvsQCD = (Xqq+Xcs)/(Xqq+Xcs+QCD)`), i.e. deterministic functions of the raw
  heads. Feeding them alongside the raw heads is collinear and will confuse a
  linear fit. Keep `TopvsQCD` only as the **baseline** to beat.
- `withMassTopvsQCD`, `withMassWvsQCD`, `withMassZvsQCD` — **mass-correlated**;
  using them re-introduces the $m_{\mathrm{SD}}$ dependence we must keep out for
  2DAlphabet. **Forbidden as inputs.**
- `massCorrGeneric`, `massCorrX2p` — mass *regression* outputs, not class scores.
  Not classifier inputs (could be used elsewhere for a regressed mass).

**IMPORTANT — the mass-decorrelated heads are NOT probabilities.** Empirically (on
real 2024 jets) the `globalParT3_*` mass-decorrelated heads do **not** sum to 1
and some (e.g. `Xqq`) **exceed 1.0**. They are unnormalized, non-negative scores,
not a softmax simplex. Consequences:

- **Do not** apply a probability logit $\log[P/(1-P)]$ — it returns NaN for $P>1$
  and assumes a normalization the data do not satisfy. Use the **log-score**
  transform of §4.1 instead.
- The sanity check Claude Code should assert is therefore **finiteness and
  non-negativity** of the 7 input heads, *not* "sum to 1". (If a separate
  *normalized* raw-softmax set is available, a sum-to-1 check may be applied to
  *that* set for monitoring only; the analysis uses the decorrelated heads.)

---

## 4. Method

### 4.1 Features

The input heads are unnormalized non-negative scores (§3), **not** probabilities,
so the probability logit is invalid. Default to a **log-score** transform: robust
to $P>1$ and $P=0$, and it Gaussianizes the multiplicative score range (good for
LDA):

$$
\phi_i = \log(P_i + \varepsilon), \qquad
P_i \in \{TopbWqq, TopbWq, QCD, Xqq, Xcs, Xbb, Xcc\}, \quad \varepsilon = 10^{-6}.
$$

This choice matters because LDA is linear in the features: a linear combination of
log-scores is the log of a product/ratio of powers, so log-likelihood-ratio
structure like $\log(P_{\rm sig}/P_{\rm QCD})$ is automatically inside the LDA
span. Provide two alternatives and compare (the fit is cheap):

- **log-ratio to QCD**, $\phi_i = \log[(P_i+\varepsilon)/(P_{QCD}+\varepsilon)]$
  (6 features; QCD is the reference). Drops the overall-scale common mode and
  matches the CMS "use sig/(sig+bkg)" guidance and the baseline's structure. This
  is the **decorrelation-safe fallback**.
- **clipped logit** (the old default): valid only on a *normalized* head set; on
  the decorrelated heads it discards the $P>1$ information — **not recommended**.

Because the log-ratio features are linear functions of the log-scores
($\log P_i - \log P_{QCD}$), by LDA linear-invariance the log-score basis
*contains* the log-ratio solution plus one extra overall-scale mode. **Default to
log-score**; if Stage 4 shows that retained scale mode re-correlates with
$m_{\mathrm{SD}}$/$p_T$, switch to log-ratio.

### 4.2 Per-$p_T$ linear discriminant from accumulated moments (PRIMARY, ntuple-free)

Why not logistic regression directly? Logistic regression mixes the heads using
their *correlations*, so it cannot be recovered from 1D marginal histograms; and
its MLE has no finite sufficient statistic — it requires iterating over per-jet
rows (i.e. ntuples). Instead use a **Gaussian linear discriminant (LDA)**, whose
solution depends only on **per-class first and second moments**, which *are*
coffea-accumulatable sums.

For each $(p_T\text{ bin}, \text{class } c)$ with $c \in \{\text{matched-top},
\text{QCD-incl}\}$, accumulate (weighted) sufficient statistics over the
length-7 feature vector $\boldsymbol\phi$ (the log-scores of §4.1):

$$
n_c=\sum w,\qquad \mathbf{S}_c=\sum w\,\boldsymbol\phi,\qquad
\mathbf{M}_c=\sum w\,\boldsymbol\phi\boldsymbol\phi^{\!\top}.
$$

These are tiny (a 7-vector + a 7×7 matrix per bin per class) and merge by addition
like any accumulator. Offline:

$$
\mu_c=\frac{\mathbf S_c}{n_c},\quad
\Sigma_c=\frac{\mathbf M_c}{n_c}-\mu_c\mu_c^{\!\top},\quad
\Sigma=\frac{n_{\rm sig}\Sigma_{\rm sig}+n_{\rm bkg}\Sigma_{\rm bkg}}{n_{\rm sig}+n_{\rm bkg}},
$$
$$
\mathbf w \propto \Sigma^{-1}(\mu_{\rm sig}-\mu_{\rm bkg}),\qquad
s=\mathbf w^{\!\top}\boldsymbol\phi + b .
$$

In log-score space, if the per-class features are ~Gaussian with similar covariance,
this LDA *is* the optimal likelihood ratio and matches logistic regression. Add a
small ridge to $\Sigma$ (`Σ + λI`) for numerical stability. An optional **QDA**
variant (keep $\Sigma_c$ per class → quadratic boundary) uses the *same* moments.
Applying a final `sigmoid(s)` is allowed (monotonic; changes neither ROC nor the
working-point inversion).

The bias $b$ is not needed for ROC/decorrelation (threshold is scanned), but fix a
convention (e.g. equal class priors) so saved scores are reproducible.

### 4.3 Optional upgrade: true logistic refit on a SMALL skim (only if LDA promising)

If §4.2 looks worth pursuing and you suspect the log-score features are
non-Gaussian, refit the ~7 weights with `sklearn` `LogisticRegression` on a
**small per-jet skim** (a few $\times10^5$ jets is ample for 7 parameters) — *not*
full production ntuples. This is the only place a per-jet table appears, and it is
deliberately small and disposable. Compare its weights/ROC to the LDA solution;
if they agree, the ntuple-free LDA is validated and stays the production path.

### 4.4 Decorrelation — the entry ticket

Because inputs are restricted to the *mass-decorrelated* heads, `s` should remain
approximately decorrelated from $m_{\mathrm{SD}}$ — but this **must be verified,
not assumed** (Stage 4). Escalation ladder if `s` sculpts mass:

1. accumulate moments **in slices of $m_{\mathrm{SD}}$** and require
   slice-independent $\mathbf w$; or
2. add a distance-correlation (DisCo) penalty against $m_{\mathrm{SD}}$ — this
   requires the §4.3 small-skim path (gradient-based fit); or
3. apply a DDT-style flattening of the final threshold vs $p_T$.

Keep v1 simple (plain per-$p_T$ LDA) and only escalate if Stage 4 fails.

---

## 5. Locked-in vs assumed decisions

Mirror the conventions of `toptag_wp_derivation_plan.md`. Items marked
**[confirm]** are assumptions Claude Code should surface, not silently bake in.

| Item | Decision |
|---|---|
| Production mode | **Pure coffea: histograms + per-class moment accumulators. No ntuples (default).** |
| Discriminant under test | Per-$p_T$ linear discriminant `s` over raw heads in §3 |
| Fit method | **LDA from accumulated moments** (NumPy, offline). Logistic refit optional on a small skim (§4.3) |
| Baseline to beat | `TopvsQCD` exactly as `_tscore` / `_topvsqcd_globalParT3` |
| Jet pre-selection | `pT > 400 GeV`, `|eta| < 2.5`, valid subjets (match WP plan) |
| Mass window | derive/evaluate inside `105 < mSD < 210`; **also** dump full-$m_{\mathrm{SD}}$ for the decorrelation check |
| $p_T$ dependence | per-$p_T$ bins (`PT_BIN_EDGES` from `toptag_wp_processor.py`) |
| Background | QCD MC (HT-stitched). Data fail region used **only** to validate decorrelation, never for fitting |
| Signal | TTbar (and/or signal) MC, AK8 matched to gen hadronic top, `dR < 0.8` (`get_hadronic_tops`) |
| Train/test split | by **event parity** (even = fit moments, odd = evaluation histograms) — no leakage, no ntuples |
| Comparison convention | fixed QCD mis-tag → compare signal eff (same as WP plan); report the dual too |
| Integration into analysis | **Not now.** Produce metrics, plots, and a saved weights file only |
| Calibration/SF | out of scope for the study; flagged as adoption cost (§11) |

---

## 6. Inputs / samples

Reuse the WP framework's samples and weighting exactly:

- **Signal jets:** TTbar MC 2024 (and optionally `RSGluon`/`Zprime` signal MC for
  high-mass coverage), AK8 jets matched within `dR < 0.8` to a gen hadronic top
  via `get_hadronic_tops` (`python/truthstudy.py:9`) /
  `_matched_to_gen_top` (`python/toptag_wp_processor.py:193`).
- **Background jets:** QCD multijet MC 2024, HT-binned, **stitched with
  xsec × lumi weights** (same as the WP plan, §3). Mirror the QCD large-genWeight
  rejection (`_qcd_genweight_mask`, `toptag_wp_processor.py:108`).
- **Data fail region (decorrelation validation only):** antitag selection from
  `TTbarResProcessor` — loose `< TopvsQCD <` medium on the second jet.
- Filesets under `data/nanoAOD/*.json`. **NanoAOD v15 required** (the
  `globalParT3_*` branches only exist from v15).

Per-jet weight = generator weight × MC normalization; carry it through all
moments and histograms (weighted).

---

## 7. Implementation plan (two-pass, pure coffea)

Pass 1 cannot know the weights (they need the *global* merged moments), so the
study is inherently two passes over the data.

### Stage 0 — Extend `TopTagWPProcessor` (no ntuples)

`TopTagWPProcessor` already emits histograms only (module docstring,
`toptag_wp_processor.py:10`). Add three things, keeping existing outputs intact:

1. **Moment accumulators** keyed by `(jettype, pt_bin)`, holding `n` (scalar),
   `S` (length-7 vector), `M` (7×7 matrix) of the log-score features. Use a coffea
   accumulator (e.g. `dict_accumulator` of `column_accumulator`/arrays) so they
   merge by addition. `jettype ∈ {matched, incl}` as already used
   (`toptag_wp_processor.py:278`).
2. A **`score_vs_mtt`** histogram twin of the existing `score_vs_msd`
   (`Hist[dataset, jettype, mtt, disc]`), where `mtt` is the invariant mass of
   the two leading AK8 jets (event context, filled per jet).
3. Extend the per-IOV `_CONFIG` (`toptag_wp_processor.py:66`) so
   `required_fields` includes `globalParT3_Xqq/Xcs/Xbb/Xcc`.

Add an **event-parity switch** so the same job routes even/odd events into
separate accumulators (fit) vs evaluation histograms (test).

### Stage 1 — Offline LDA fit from moments (tiny NumPy)

Library `python/glopart_recomb.py`:

```python
RAW_HEADS = ["TopbWqq","TopbWq","QCD","Xqq","Xcs","Xbb","Xcc"]

def logscore(p, eps=1e-6):                  # heads are scores, not probs (may exceed 1)
    return np.log(np.clip(p, 0.0, None) + eps)

def baseline_topvsqcd(df):                  # the score to beat
    num = df.TopbWqq + df.TopbWq
    return num / (num + df.QCD)

def moments_to_lda(n_s, S_s, M_s, n_b, S_b, M_b, ridge=1e-4):
    mu_s, mu_b = S_s/n_s, S_b/n_b
    cov = lambda n,S,M: M/n - np.outer(S/n, S/n)
    Sig = (n_s*cov(n_s,S_s,M_s) + n_b*cov(n_b,S_b,M_b)) / (n_s+n_b)
    Sig += ridge*np.eye(len(mu_s))
    w = np.linalg.solve(Sig, mu_s - mu_b)
    return w

def fit_per_pt(moments):                    # moments[(jettype,pt_bin)] -> {pt_bin: w}
    return {pt: moments_to_lda(*sig_and_bkg(moments, pt)) for pt in pt_bins(moments)}
```

Persist `{pt_edges, RAW_HEADS, w_per_bin, bias_convention, metadata}` to
`outputs/glopart_recomb/weights_2024.json` (portable; no `pickle`).

### Stage 2 — Pass 2: fill score histograms for `s`

Re-run the processor in **apply mode**: load `weights_2024.json`, swap the
`cfg['score']` function from `_topvsqcd_globalParT3` to
`s = w[pt_bin] · logscore(heads)`, and fill `score`, `score_vs_msd`, `score_vs_mtt`
on the **test** (odd) parity. The baseline histograms come from Pass 1.

### Stage 3 — Fixed-mis-tag ROC comparison (WP-plan convention)

For each $p_T$ bin and each target QCD mis-tag (WP-plan targets
0.1/0.5/1.0/2.5/5.0%): invert baseline and `s` on test QCD (`incl`) to the
threshold giving that mis-tag, then report each one's **signal efficiency**
(`matched`). Higher sig-eff at equal mis-tag = win. Report the dual (fixed
sig-eff → mis-tag) to connect to the current medium WP (0.8571).

### Stage 4 — Decorrelation & mis-tag flatness (DECIDING METRIC)

All from histograms:

- **Mis-tag vs $m_{t\bar t}$**: from `score_vs_mtt[jettype=incl]`, in each
  $m_{t\bar t}$ slice compute the fraction above the matched-mis-tag threshold.
  Fit the resulting `mistag(mtt)` curve to a constant; smaller slope / $\chi^2$
  for `s` ⇒ more factorizable background ⇒ TF order can drop.
- **Mass sculpting**: repeat with `score_vs_msd` for `mistag(msd)`; also report a
  binned correlation between disc and `msd` on QCD. Require `s` ≤ baseline.
- Validate on **QCD MC** and the **data fail region**.

### Stage 5 — Optional: logistic cross-check on a small skim (§4.3)

Only if Stage 4 is promising and Gaussianity is in doubt. Small disposable skim,
not production ntuples.

### Stage 6 — Optional downstream confirmation (heavy)

Regenerate the 2DAlphabet pass/fail inputs with `s` as the tag and re-run
`fit_ftest.py`. A lower selected TF order than `2x1` is the strongest adoption
evidence.

---

## 8. Suggested file layout

```
python/glopart_recomb.py             # logscore, baseline, moments->LDA, apply, metrics (library)
run_glopart_recomb.py                # driver: pass1 (moments+baseline hists) -> fit -> pass2 (s hists) -> metrics
plot_glopart_recomb.py               # ROC, mistag(mtt), mistag(msd), weight tables
outputs/glopart_recomb/
    moments_2024.coffea              # accumulated per-(jettype,pt_bin) n,S,M  (pass 1)
    hists_baseline_2024.coffea       # baseline score hists                    (pass 1)
    weights_2024.json                # fitted per-pt LDA weights               (offline)
    hists_recomb_2024.coffea         # s score hists                           (pass 2)
    metrics_2024.json                # ROC + flatness numbers
    SUMMARY.md                       # verdict vs §2 decision rule
plots/glopart_recomb/2024/*.png
tests/test_glopart_recomb.py
# (optional, Stage 5 only) outputs/glopart_recomb/skim_2024.parquet
```

Reuse `run_toptag_wp.py` / `plot_toptag_wp.py` as structural templates.

---

## 9. Tests Claude Code should write (mirror existing `tests/`)

Follow the style of `tests/test_toptag_wp_processor.py`,
`test_run_toptag_wp.py`, `test_plot_toptag_wp.py`. Use tiny synthetic awkward/NumPy
fixtures; **no grid/xrootd access in unit tests.**

1. **`test_heads_finite_nonneg`** — input heads are finite and ≥ 0 on the fixture;
   a NaN/negative head is flagged. **Do not** assert sum-to-1: the decorrelated
   heads are not a simplex and `Xqq` can exceed 1.
2. **`test_logscore_transform`** — `logscore(P)` is finite and has no NaN/inf for
   `P=0`, `P>1`, and large `P`, and is monotonic in `P` (confirms the probability
   logit is correctly *not* used).
3. **`test_baseline_matches_tscore`** — `baseline_topvsqcd` equals
   `_tscore` / `_topvsqcd_globalParT3` on the same inputs (guards drift).
4. **`test_moment_accumulation_merges`** — accumulating `n,S,M` over chunks then
   merging equals direct computation on the full set (the core ntuple-free
   correctness test).
5. **`test_lda_recovers_separable`** — on synthetic Gaussian classes with known
   $\mu,\Sigma$, `moments_to_lda` recovers $w \propto \Sigma^{-1}\Delta\mu$ and
   gives high AUC; identical classes give AUC ≈ 0.5 (no leakage).
6. **`test_per_pt_routing`** — each jet scored with its own $p_T$-bin weights; no
   NaN; edge $p_T$ values land in the right bin.
7. **`test_fixed_mistag_efficiency`** — threshold inverted from the score
   histogram reproduces the target mis-tag on held-out QCD within Poisson
   tolerance.
8. **`test_train_test_parity_split`** — even events feed moments, odd events feed
   evaluation histograms; both jets of an event share parity; no overlap.
9. **`test_decorrelation_from_hist`** — `mistag(mtt)` read from
   `Hist[jettype,mtt,disc]` matches a direct per-jet computation on a fixture; a
   constructed mass-correlated score is non-flat, a mass-flat one is flat (metric
   direction sanity).
10. **`test_weights_roundtrip`** — save/load `weights_2024.json` reproduces
    scores bit-for-bit.
11. **`test_processor_accumulates_required_heads`** — extended processor output
    contains the moment accumulators (right shapes: scalar `n`, 7-vector `S`,
    7×7 `M`) and the `score_vs_mtt` histogram, for all heads in §3.

---

## 10. Acceptance criteria

The study is "done" (independently of the physics verdict) when:

- Pass 1 produces, for 2024 signal + QCD MC, the per-`(jettype,pt_bin)` moment
  accumulators and baseline score histograms — **no ntuples**;
- the head finiteness / non-negativity check passes on real jets (no sum-to-1
  assumption);
- per-$p_T$ LDA weights are fit from the moments, saved, and reloadable;
  train/test split is by event parity;
- Pass 2 fills the `s` score histograms on the test parity;
- Stage-3 table reports signal-eff (baseline vs `s`) at each target mis-tag and
  $p_T$ bin, test set only;
- Stage-4 reports mis-tag flatness vs $m_{t\bar t}$ and $m_{\mathrm{SD}}$ (slope,
  $\chi^2$) and the disc–mass correlation for baseline vs `s`, on QCD MC and data
  fail region;
- all §9 tests pass;
- `outputs/glopart_recomb/SUMMARY.md` states the verdict against §2's rule.

---

## 11. Risks and notes

- **LDA optimality assumption.** LDA = logistic only when the per-class log-score
  features are ~Gaussian with similar covariance. If strongly non-Gaussian, LDA
  is sub-optimal; mitigations are QDA (same moments) or the §4.3 small-skim
  logistic refit. Neither requires production ntuples.
- **Calibration ownership (the main adoption cost).** A custom score has **no
  central scale factor**. Adopting `s` means deriving its own top-tag SF in a
  $t\bar t$-enriched region, on top of the GloParTv3 SF already on the critical
  path. The study does not need this; adoption does.
- **Expected ROC gain is modest** (low single-digit % rejection) — `TopvsQCD` is
  already near-optimal for top-vs-QCD. Adoption rests on Stage 4/6 (flattening →
  lower TF order), not Stage 3.
- **Do not leak the blinded signal window** into the fit. Fit on MC moments; use
  data only for the decorrelation check, outside the blinded pass window.
- **The decorrelated heads are not probabilities** (unnormalized, can exceed 1).
  Use `log(P+ε)`, never the probability logit. Choice of transform (log-score vs
  log-ratio-to-QCD) matters; default log-score, fall back to log-ratio if the
  scale mode re-correlates (§4.1).
- **Numerical care**: ridge-regularize $\Sigma$; assert head finiteness /
  non-negativity; never assume sum-to-1.
- Keep all inputs to the mass-decorrelated heads; never `withMass*`/`massCorr*`.

---

## 12. References

- Baldi, Cranmer, Faucett, Sadowski, Whiteson, *Parameterized neural networks for
  high-energy physics*, arXiv:1601.07913 (context for the Tier-1 follow-on).
- Dolen et al., *Designing Decorrelated Taggers (DDT)*, arXiv:1603.00027.
- Kasieczka, Shih, *DisCo Fever: Robust Networks Through Distance Correlation*,
  arXiv:2001.05310 (escalation option in §4.4).
- Internal: `toptag_wp_derivation_plan.md` (shared samples + mis-tag convention);
  `ttbarprocessor.py:274` (`_tscore`); `python/toptag_wp_processor.py`
  (`TopTagWPProcessor`); `python/truthstudy.py` (gen matching).

---

*This is a study/test plan. It does not modify the analysis selection, the
2DAlphabet model, or `_tscore`. Adoption into the analysis is a separate decision
gated by §2 and §11.*
