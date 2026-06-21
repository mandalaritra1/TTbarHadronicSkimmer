"""Coffea processor for deriving GloParTv3 top-tagging working points.

This processor is intentionally separate from ``TTbarResProcessor``: it does NOT
run the full all-hadronic event selection. Instead it fills per-AK8-jet
histograms of the top-tagger discriminant, which is exactly the population
needed to derive working points (the WP threshold is defined by the QCD
background mis-tag efficiency, and the signal efficiency is read off the same
distribution from gen-matched tops in TTbar).

No ntuples are produced — only histograms.

Year handling is data-driven via ``TAGGER_CONFIG`` so that adding a new IOV
later (once NanoAOD v15 is available for it) is a one-line change. There is no
hard-coded ``'2024'`` in the logic.

Outputs (keys in the accumulator dict):
  - ``score``        : Hist[dataset, jettype, pt, disc] inside the mass window.
                       Used to derive pT-binned WP thresholds and signal eff.
  - ``score_full_msd`` : Hist[dataset, jettype, pt, disc] without the mass window.
                       Used for Data/MC score-shape checks including sidebands.
  - ``score_vs_msd`` : Hist[dataset, jettype, msd, disc] WITHOUT the mass window.
                       Used for the mass-decorrelation cross-check.
  - ``jet_pt``       : Hist[dataset, jettype, pt] for preselected AK8 jets.
                       Used as a QCD stitching / smoothness control plot.
  - ``sumw``         : defaultdict(float), sum of genWeight per dataset.
  - ``nevents``      : defaultdict(int),   events kept after preprocessing.
  - ``nevents_raw``  : defaultdict(int),   input events before preprocessing.
  - ``qcd_genweight_rejected`` : defaultdict(int), QCD events removed by the
                       large-genWeight filter mirrored from ``TTbarResProcessor``.

``jettype`` is ``"incl"`` (all preselected jets) or ``"matched"`` (AK8 matched
to a gen hadronic top, filled only for signal/TTbar samples). The background
(QCD) mis-tag uses ``"incl"``; the signal efficiency uses ``"matched"``.
"""

import os
import sys

import numpy as np
import awkward as ak
import hist
from coffea import processor

# Mirror the import convention used by ttbarprocessor.py so this works both
# locally and on Dask workers (where the package path is not importable).
sys.path.append(os.getcwd() + '/python/')
from truthstudy import get_hadronic_tops, _ensure_p4  # noqa: E402
import glopart_recomb as gr  # noqa: E402


# ---------------------------------------------------------------------------
# Per-IOV tagger configuration. To add a year, add an entry here with a
# discriminant function and the NanoAOD FatJet fields it requires.
# ---------------------------------------------------------------------------
def _topvsqcd_globalParT3(fatjets):
    """GloParTv3 mass-decorrelated TopvsQCD discriminant.

    (TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD)

    Identical to TTbarResProcessor._tscore so derived WPs match the analysis.
    """
    num = fatjets.globalParT3_TopbWqq + fatjets.globalParT3_TopbWq
    den = num + fatjets.globalParT3_QCD
    return num / den


TAGGER_CONFIG = {
    '2024': {
        'score': _topvsqcd_globalParT3,
        'required_fields': [
            'globalParT3_TopbWqq',
            'globalParT3_TopbWq',
            'globalParT3_QCD',
        ],
        'label': 'globalParT3_TopvsQCD',
    },
    # Add future IOVs once NanoAOD v15 exists for them, e.g.:
    # '2025': {'score': _topvsqcd_globalParT3,
    #          'required_fields': [...], 'label': 'globalParT3_TopvsQCD'},
}

# pT bin edges used for the pT-dependent WP derivation (GeV). The last edge is a
# finite stand-in for "infinity".
PT_BIN_EDGES = [400.0, 500.0, 600.0, 800.0, 1200.0, 3000.0]

# Number of fine discriminant bins. A fine binning is required so the cumulative
# tail used to invert for a target mis-tag rate is smooth.
N_DISC_BINS = 1000

# Fine-but-readable pT control bins. Keep the threshold/stitching-sensitive
# region reasonably granular, then open up the sparse high-pT tail.
PT_CONTROL_EDGES = np.concatenate([
    np.arange(400.0, 1200.0, 20.0),
    np.arange(1200.0, 3000.0 + 100.0, 100.0),
])

# Mass window (GeV) applied when deriving WPs (kept, per analysis convention).
MSD_MIN = 105.0
MSD_MAX = 210.0

# Jet pre-selection.
JET_PT_MIN = 400.0
JET_ETA_MAX = 2.5

# gen-top match radius.
DR_MATCH = 0.8

# --- Recomb-study additions (opt-in via TopTagWPProcessor(recomb_study=True)) ---
# The 4 extra two-prong heads needed by the recombination, beyond the 3 the
# baseline TopvsQCD ratio already uses. Together with the baseline's 3 heads
# these form gr.RAW_HEADS.
RECOMB_EXTRA_HEADS = [
    'globalParT3_Xqq', 'globalParT3_Xcs', 'globalParT3_Xbb', 'globalParT3_Xcc',
]
# Invariant mass of the two leading AK8 jets (event context), used for the
# mis-tag-vs-mtt decorrelation check.
MTT_EDGES = (70, 0.0, 7000.0)

# --- Ntuple mode (opt-in via TopTagWPProcessor(recomb_ntuple=True)) ---
# A *skinny* per-jet ntuple holding ONLY the columns the recombination fit needs,
# so a full-statistics Dask run (coffea.casa) can feed the data-space LOGISTIC fit
# (moments only give LDA). Columns mirror data/skim/*.npz plus mtt + label.
#   label: 1 = gen-matched hadronic top (signal), 0 = inclusive QCD (background)
#   genweight stored RAW; xsec*lumi/sumw normalization is applied at save time.
NTUPLE_FIELDS = {
    **{h: np.float32 for h in gr.RAW_HEADS},   # 7 GloParTv3 heads
    'pt': np.float32, 'msd': np.float32, 'mtt': np.float32,
    'genweight': np.float32, 'parity': np.int8, 'label': np.int8,
}

# Dataset-name prefixes treated as SIGNAL (fill the gen-matched-top "matched"
# class). TTbar for the bulk; resonance MC (Z'/RS graviton) for the boosted tail.
SIGNAL_NAME_PREFIXES = ('TT', 'ZPRIME', 'ZPTOTT', 'RSG', 'RSGLUON')


def _qcd_genweight_mask(gen_weight, nsigma=2.0):
    """Mirror TTbarResProcessor's QCD large-genWeight event rejection."""
    vals = ak.to_numpy(gen_weight)
    if len(vals) == 0:
        return ak.ones_like(gen_weight, dtype=bool)

    average = np.average(vals)
    stddev = np.std(vals)
    if stddev == 0 or not np.isfinite(stddev):
        return ak.ones_like(gen_weight, dtype=bool)

    return np.abs((gen_weight - average) / stddev) < nsigma


def _make_score_hist():
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="jettype", growth=True),
        hist.axis.Variable(PT_BIN_EDGES, name="pt", label=r"AK8 $p_T$ [GeV]"),
        hist.axis.Regular(N_DISC_BINS, 0.0, 1.0, name="disc", label="TopvsQCD"),
        storage="weight",
        name="Counts",
    )


def _make_score_full_msd_hist():
    return _make_score_hist()


def _make_score_vs_msd_hist():
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="jettype", growth=True),
        hist.axis.Regular(60, 0.0, 300.0, name="msd", label=r"$m_{SD}$ [GeV]"),
        hist.axis.Regular(200, 0.0, 1.0, name="disc", label="TopvsQCD"),
        storage="weight",
        name="Counts",
    )


def _make_score_vs_mtt_hist():
    """score vs invariant mass of the two leading AK8 jets (recomb study only)."""
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="jettype", growth=True),
        hist.axis.Regular(*MTT_EDGES, name="mtt", label=r"$m_{t\bar t}$ [GeV]"),
        hist.axis.Regular(200, 0.0, 1.0, name="disc", label="disc"),
        storage="weight",
        name="Counts",
    )


def _make_jet_pt_hist():
    return hist.Hist(
        hist.axis.StrCategory([], name="dataset", growth=True),
        hist.axis.StrCategory([], name="jettype", growth=True),
        hist.axis.Variable(PT_CONTROL_EDGES, name="pt", label=r"AK8 $p_T$ [GeV]"),
        storage="weight",
        name="Counts",
    )


class TopTagWPProcessor(processor.ProcessorABC):
    """Fill per-AK8-jet tagger-discriminant histograms for WP derivation.

    Parameters
    ----------
    iov : str
        IOV key into ``TAGGER_CONFIG`` (e.g. ``'2024'``). No year is hard-coded
        in the logic; an unknown IOV raises immediately.
    match_gen_top : bool or None
        If True, also fill the ``"matched"`` jettype (AK8 matched to a gen
        hadronic top). If None (default), auto-detect from sample metadata /
        dataset name (samples starting with ``TT``).
    sample_metadata : dict or None
        Optional per-sample metadata (sample, subsample, year, is_mc, xsec_pb).
    """

    def __init__(self, iov='2024', match_gen_top=None, sample_metadata=None,
                 recomb_study=False, recomb_weights=None, recomb_transform='logscore',
                 recomb_ntuple=False, recomb_ntuple_pt_min=None, recomb_ntuple_prescale=1.0,
                 recomb_ntuple_prescale_below=None):
        if iov not in TAGGER_CONFIG:
            raise KeyError(
                f"IOV '{iov}' not in TAGGER_CONFIG (known: {list(TAGGER_CONFIG)}). "
                "Add an entry with a discriminant + required NanoAOD fields."
            )
        self.iov = iov
        self._cfg = TAGGER_CONFIG[iov]
        self.match_gen_top = match_gen_top
        self.sample_metadata = dict(sample_metadata or {})

        # --- recomb study config (all inert unless recomb_study=True) ---
        # recomb_study   : turn on moment accumulation + score_vs_mtt + parity split
        # recomb_weights : None  -> pass 1 (accumulate moments + baseline eval hists)
        #                  path/dict -> pass 2 (fill recombination-s eval hists)
        # recomb_transform : feature transform name in gr.VALID_TRANSFORMS
        self.recomb_study = bool(recomb_study)
        self.recomb_ntuple = bool(recomb_ntuple)
        # Memory controls for the per-jet ntuple (the only growth-with-data output):
        #   pt_min        : hard-drop jets below this pT (default = preselection
        #                   floor, i.e. no cut — keeps the full analysis pT range).
        #   prescale      : keep this fraction of jets, reweighting by 1/frac.
        #   prescale_below: if set, the prescale applies ONLY to jets below this pT
        #                   (all jets above are kept). This thins the abundant
        #                   low-pT QCD bulk while preserving the sparse high-pT tail
        #                   that drives the 0.5% mis-tag — the memory-safe way to
        #                   keep the full pT range. None => prescale applies to all.
        self.recomb_ntuple_pt_min = (JET_PT_MIN if recomb_ntuple_pt_min is None
                                     else float(recomb_ntuple_pt_min))
        self.recomb_ntuple_prescale = float(recomb_ntuple_prescale)
        self.recomb_ntuple_prescale_below = (None if recomb_ntuple_prescale_below is None
                                             else float(recomb_ntuple_prescale_below))
        self.recomb_transform = recomb_transform
        if isinstance(recomb_weights, str):
            recomb_weights = gr.load_weights(recomb_weights)
        self.recomb_weights = recomb_weights

    # -- helpers ------------------------------------------------------------
    def _should_match(self, dataset):
        if self.match_gen_top is not None:
            return bool(self.match_gen_top)
        sample = str(self.sample_metadata.get('sample', '')).upper()
        ds = str(dataset).upper()
        # Signal = TTbar OR resonance MC (Z'/RS graviton) -> has gen hadronic tops.
        return (sample.startswith(SIGNAL_NAME_PREFIXES)
                or ds.startswith(SIGNAL_NAME_PREFIXES))

    @staticmethod
    def _matched_to_gen_top(events):
        """Return a per-jet boolean: AK8 matched to a gen hadronic top (dR<0.8)."""
        tops = _ensure_p4(get_hadronic_tops(events.GenPart))
        fatjets = _ensure_p4(events.FatJet)
        pairs = ak.cartesian({"fat": fatjets, "top": tops}, axis=1, nested=True)
        dr = pairs["fat"].p4.delta_r(pairs["top"].p4)
        min_dr = ak.fill_none(ak.min(dr, axis=2), 999.0)
        return min_dr < DR_MATCH

    def _recomb_disc(self, fj):
        """Pass-2 recombination score s (sigmoid-squashed to [0,1]) per jet.

        Returns a jagged array aligned with ``fj`` so the existing fill helpers
        work unchanged. Each jet is scored with the LDA weights of its own pT
        bin (``gr.apply_per_pt``).
        """
        counts = ak.num(fj, axis=1)
        cols = {h: ak.to_numpy(ak.flatten(fj["globalParT3_" + h])) for h in gr.RAW_HEADS}
        X = gr.build_features(cols, transform=self.recomb_transform)
        pt_flat = ak.to_numpy(ak.flatten(fj.pt))
        s = gr.apply_per_pt(
            X, pt_flat,
            self.recomb_weights["pt_edges"],
            self.recomb_weights["weights_per_bin"],
            sigmoid=True,
        )
        return ak.unflatten(s, ak.to_numpy(counts))

    # -- coffea API ---------------------------------------------------------
    def process(self, events):
        dataset = events.metadata['dataset']
        cfg = self._cfg
        recomb = self.recomb_study
        ntuple = self.recomb_ntuple
        need_ctx = recomb or ntuple              # need parity / mtt / extra heads
        apply_mode = recomb and self.recomb_weights is not None  # pass 2

        output = {
            'score': _make_score_hist(),
            'score_full_msd': _make_score_full_msd_hist(),
            'score_vs_msd': _make_score_vs_msd_hist(),
            'jet_pt': _make_jet_pt_hist(),
            'sumw': processor.defaultdict_accumulator(float),
            'nevents': processor.defaultdict_accumulator(int),
            'nevents_raw': processor.defaultdict_accumulator(int),
            'qcd_genweight_rejected': processor.defaultdict_accumulator(int),
        }
        if recomb:
            output['score_vs_mtt'] = _make_score_vs_mtt_hist()
            # per-(jettype, pt_bin) packed moment vectors; merge by addition.
            output['moments'] = processor.defaultdict_accumulator(gr.empty_moment_vec)
        if ntuple:
            # nested {dataset -> {field -> column_accumulator}}; coffea merges by
            # concatenation across chunks and keeps datasets separate for per-sample
            # cross-section weighting at save time.
            output['ntuple'] = {}

        n_raw = len(events)
        n_rejected = 0
        if 'QCD' in dataset and 'genWeight' in events.fields:
            genweight_mask = _qcd_genweight_mask(events.genWeight)
            n_rejected = n_raw - int(ak.sum(genweight_mask))
            events = events[genweight_mask]

        fj = events.FatJet
        required = list(cfg['required_fields'])
        if need_ctx:
            required = required + RECOMB_EXTRA_HEADS
        missing = [f for f in required if f not in fj.fields]
        if missing:
            raise RuntimeError(
                f"FatJet missing {missing} for IOV {self.iov} (dataset {dataset}). "
                "Confirm the sample is NanoAOD v15 with GloParTv3 branches."
            )

        # Per-event weight (genWeight for MC, 1.0 for data / when absent).
        if 'genWeight' in events.fields:
            w_evt = events.genWeight
        else:
            w_evt = ak.ones_like(ak.num(fj, axis=1), dtype=np.float64) * 1.0

        output['sumw'][dataset] += float(ak.sum(w_evt))
        output['nevents'][dataset] += int(len(events))
        output['nevents_raw'][dataset] += int(n_raw)
        output['qcd_genweight_rejected'][dataset] += int(n_rejected)

        # Discriminant filled into the eval histograms: baseline TopvsQCD in
        # pass 1 (and the non-recomb WP path), recombination s in pass 2.
        disc = self._recomb_disc(fj) if apply_mode else cfg['score'](fj)
        pt = fj.pt
        eta = fj.eta
        msd = fj.msoftdrop

        presel = (pt > JET_PT_MIN) & (abs(eta) < JET_ETA_MAX)
        window = presel & (msd > MSD_MIN) & (msd < MSD_MAX)

        # broadcast event weight to per-jet
        w_jet = ak.broadcast_arrays(w_evt, pt)[0]

        # Event-parity gating. Parity is per-event (uses the NanoAOD event number)
        # so both jets of an event share it: no leakage. even -> moment fit;
        # odd -> evaluation histograms. The ntuple stores BOTH parities (as a
        # column) so the downstream fit owns the split.
        gate = None
        parity_jet = None
        if need_ctx:
            parity = ak.values_astype(events.event, np.int64) % 2
            parity_jet = ak.broadcast_arrays(parity, pt)[0]
        if recomb:
            even_jet = ak.broadcast_arrays(parity == 0, pt)[0]
            gate = ak.broadcast_arrays(parity == 1, pt)[0]  # odd = eval/test

        def _g(mask):
            return mask if gate is None else (mask & gate)

        def _fill(h, jettype, mask, extra_axes):
            sel_disc = ak.to_numpy(ak.flatten(disc[mask]))
            sel_w = ak.to_numpy(ak.flatten(w_jet[mask]))
            kw = {name: ak.to_numpy(ak.flatten(arr[mask])) for name, arr in extra_axes.items()}
            h.fill(dataset=dataset, jettype=jettype, disc=sel_disc, weight=sel_w, **kw)

        def _fill_pt(h, jettype, mask):
            h.fill(
                dataset=dataset,
                jettype=jettype,
                pt=ak.to_numpy(ak.flatten(pt[mask])),
                weight=ak.to_numpy(ak.flatten(w_jet[mask])),
            )

        # mtt = invariant mass of the two leading AK8 jets (NaN if <2 jets).
        # Needed by both the score-vs-mtt hist (recomb) and the ntuple.
        mtt_jet = None
        if need_ctx:
            fj2 = ak.pad_none(fj, 2, clip=True)
            mtt_evt = ak.fill_none((fj2[:, 0] + fj2[:, 1]).mass, np.nan)
            mtt_jet = ak.broadcast_arrays(mtt_evt, pt)[0]

        if recomb:
            def _fill_mtt(jettype, mask):
                d = ak.to_numpy(ak.flatten(disc[mask]))
                m = ak.to_numpy(ak.flatten(mtt_jet[mask]))
                wv = ak.to_numpy(ak.flatten(w_jet[mask]))
                good = np.isfinite(m) & np.isfinite(d)
                output['score_vs_mtt'].fill(
                    dataset=dataset, jettype=jettype,
                    mtt=m[good], disc=d[good], weight=wv[good],
                )

        def _accumulate_moments(jettype, mask):
            cols = {h: ak.to_numpy(ak.flatten(fj["globalParT3_" + h][mask]))
                    for h in gr.RAW_HEADS}
            if len(cols['QCD']) == 0:
                return
            X = gr.build_features(cols, transform=self.recomb_transform)
            pt_sel = ak.to_numpy(ak.flatten(pt[mask]))
            w_sel = ak.to_numpy(ak.flatten(w_jet[mask]))
            idx = gr.pt_bin_index(pt_sel, PT_BIN_EDGES)
            for pb in range(len(PT_BIN_EDGES) - 1):
                m = idx == pb
                if np.any(m):
                    output['moments'][gr.moment_key(jettype, pb)] += gr.moment_vec(X[m], w_sel[m])

        def _accumulate_ntuple(label_val, mask):
            """Append the skinny per-jet fit columns for jets passing ``mask``.

            Stores BOTH parities and NO mass window (preselection only) so the
            downstream fit owns the parity split and any m_SD window. ``genweight``
            is raw (×1/prescale if prescaled); cross-section normalization happens
            at save time. The pT floor + prescale keep this — the only output that
            grows with data — within worker memory on full-stats runs.
            """
            flat = lambda arr: ak.to_numpy(ak.flatten(arr[mask]))
            cols = {h: flat(fj["globalParT3_" + h]).astype(np.float32) for h in gr.RAW_HEADS}
            n = len(cols['QCD'])
            if n == 0:
                return
            cols['pt'] = flat(pt).astype(np.float32)
            cols['msd'] = flat(msd).astype(np.float32)
            cols['mtt'] = flat(mtt_jet).astype(np.float32)
            cols['genweight'] = flat(w_jet).astype(np.float32)
            cols['parity'] = flat(parity_jet).astype(np.int8)
            cols['label'] = np.full(n, label_val, dtype=np.int8)
            # prescale: keep each jet with prob keep_prob, compensate weight by
            # 1/keep_prob (unbiased). If prescale_below is set, only jets under that
            # pT are thinned (keep_prob=1 above) so the high-pT tail is untouched.
            p = self.recomb_ntuple_prescale
            if p < 1.0:
                knee = self.recomb_ntuple_prescale_below
                if knee is None:
                    keep_prob = np.full(n, p, dtype=np.float64)
                else:
                    keep_prob = np.where(cols['pt'] < knee, p, 1.0)
                keep = np.random.random(n) < keep_prob
                if not np.any(keep):
                    return
                comp = (1.0 / keep_prob[keep]).astype(np.float32)
                cols = {k: v[keep] for k, v in cols.items()}
                cols['genweight'] = (cols['genweight'] * comp).astype(np.float32)
            dest = output['ntuple'].setdefault(
                dataset,
                {f: processor.column_accumulator(np.empty(0, dtype=dt))
                 for f, dt in NTUPLE_FIELDS.items()})
            for f in NTUPLE_FIELDS:
                dest[f] = dest[f] + processor.column_accumulator(cols[f])

        is_signal = self._should_match(dataset)
        do_match = is_signal and 'GenPart' in events.fields
        is_matched = self._matched_to_gen_top(events) if do_match else None

        # ---- evaluation histograms (odd/test parity in recomb mode) ----
        _fill(output['score'], 'incl', _g(window), {'pt': pt})
        _fill(output['score_full_msd'], 'incl', _g(presel), {'pt': pt})
        _fill(output['score_vs_msd'], 'incl', _g(presel), {'msd': msd})
        _fill_pt(output['jet_pt'], 'incl', _g(presel))
        if recomb:
            _fill_mtt('incl', _g(presel))

        if do_match:
            _fill(output['score'], 'matched', _g(window & is_matched), {'pt': pt})
            _fill(output['score_full_msd'], 'matched', _g(presel & is_matched), {'pt': pt})
            _fill(output['score_vs_msd'], 'matched', _g(presel & is_matched), {'msd': msd})
            _fill_pt(output['jet_pt'], 'matched', _g(presel & is_matched))
            if recomb:
                _fill_mtt('matched', _g(presel & is_matched))

        # ---- moment accumulation (even/fit parity; pass 1 only) ----
        # Class gating avoids leakage: 'matched' moments come ONLY from signal
        # (TTbar), 'incl' (background) moments come ONLY from non-signal (QCD),
        # so summing across datasets never mixes signal into the background.
        if recomb and not apply_mode:
            if do_match:
                _accumulate_moments('matched', (window & is_matched) & even_jet)
            elif not is_signal:
                _accumulate_moments('incl', window & even_jet)

        # ---- skinny ntuple (both parities, presel only; same class gating) ----
        # matched tops -> label 1 (from signal); inclusive QCD -> label 0. No mass
        # window so the fit can apply its own; no parity gate so the fit splits. A
        # pT floor drops the low-pT bulk we never fit (key memory control).
        if ntuple:
            nt_presel = presel
            if self.recomb_ntuple_pt_min > JET_PT_MIN:
                nt_presel = presel & (pt >= self.recomb_ntuple_pt_min)
            if do_match:
                _accumulate_ntuple(1, nt_presel & is_matched)
            elif not is_signal:
                _accumulate_ntuple(0, nt_presel)

        return output

    def postprocess(self, accumulator):
        return accumulator
