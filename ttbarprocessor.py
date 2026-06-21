#!/usr/bin/env python
# coding: utf-8

from coffea import processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
import hashlib
import sys
import os, psutil
import copy
import shutil
import subprocess
import tempfile
import hist
import scipy.stats as ss
import numpy as np
import pandas as pd
from numpy.random import RandomState
import random
import correctionlib
import json
import logging
import psutil
import time
import warnings

import awkward as ak

# for dask, `from python.corrections import` does not work
sys.path.append(os.getcwd() + '/python/')

from corrections import (
    GetFlavorEfficiency,
    getLumiMask,
    getMETFilter,
)
from btagCorrections import btagCorrections
from functions import getRapidity
from categories import build_analysis_categories
from cutflow import cutflow_step_metadata
from jets import Run3JetManager, _AK4_PT_MIN, _AK4_ETA_MAX


def _is_xrootd_path(path):
    return str(path).startswith("root://")


def _xrootd_url_parts(path):
    prefix, remote_path = str(path).split("//", 1)
    host, store_path = remote_path.split("/", 1)
    return f"{prefix}//{host}", f"/{store_path.lstrip('/')}"


def _xrootd_parent(path):
    server, store_path = _xrootd_url_parts(path)
    return server, os.path.dirname(store_path)


def _copy_to_xrootd(local_path, remote_path):
    server, remote_dir = _xrootd_parent(remote_path)
    subprocess.run(["xrdfs", server, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["xrdcp", "-f", local_path, remote_path], check=True)
from hists import build_output_histograms, ntuple_columns_for_preset
from weights import Run3WeightManager
import glopart_recomb as gr
from truthstudy import truthstudy_counts, build_gen_top_match_info, build_top_aligned_genjetak8_match_info


logger = logging.getLogger('__main__')
logger.setLevel(logging.DEBUG)

ak.behavior.update(vector.behavior)

# --- analysis-level constants ---
_QCD_BINVAR_MIN  = 400   # GeV — QCD HT bin lower edge; events below are below-threshold
_DPHI_CUT        = 2.1   # min |Δφ| between the two leading FatJets (back-to-back topology)
_DR_AK8          = 0.8   # standard AK8 cone radius used for dR matching
_DR_AK4          = 1.2   # dR cone for AK4 jets "near" a top (larger than the AK8 cone)
_DR_NEARBY_INNER = 0.4   # inner radius of the AK4-near-AK8 annulus

_LUMI_PB = {
    '2016APV': 19800.,
    '2016':    16120.,
    '2016all': 35920.,
    '2017':    41530.,
    '2018':    59740.,
    '2023':    27000.,
    '2024':    109950.,  # golden-JSON certified 2024 lumi (109.95 fb^-1)
}

# Per-IOV top-tagger score thresholds.
# Run-3 uses globalParT3; these keys are historically called "deepAK8" in the code.
_TAGGER_WPS = {
    'loose': {
        '2022': 0.435,
        '2023': 0.435,
        '2024': 0.6488,
        '2025': 0.470,
    },
    'medium': {
        '2022': 0.632,
        '2023': 0.632,
        '2024': 0.8571,
        '2025': 0.685,
    },
    'tight': {
        '2016APV': 0.889,
        '2016':    0.889,
        '2017':    0.863,
        '2018':    0.920,
        '2024':    0.9284,
    },
}


def get_memory_usage(human_readable=True, precision=2):
    process = psutil.Process(os.getpid())
    memory_usage_bytes = process.memory_info().rss

    if not human_readable:
        return memory_usage_bytes / (1024 * 1024)

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(memory_usage_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    return f"{size:.{precision}f} {units[unit_index]}"


class Logger:
    DEBUG = 10
    INFO = 20

    def __init__(self, mode='debug'):
        self.level = self.DEBUG if mode == 'debug' else self.INFO

    def debug(self, msg, *args):
        if self.level <= self.DEBUG:
            print('[DEBUG]', msg % args if args else msg)

    def info(self, msg, *args):
        if self.level <= self.INFO:
            print('[INFO]', msg % args if args else msg)


def update(events, collections):
    """Return a shallow copy of events with some collections swapped out."""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class TTbarResProcessor(processor.ProcessorABC):
    """Coffea processor for the all-hadronic ttbar resonance search.

    Applies trigger, kinematic, and top-tagging selections to NanoAOD events,
    then fills histograms (and optionally a flat ntuple) for each analysis
    category and systematic variation.
    """

    def __init__(
        self,
        htCut=1500,
        ak8PtMin=400.,
        minMSD=105.,
        maxMSD=210.,
        tau32Cut=0.65,
        bdisc=0.5847,
        deepAK8Cut='medium',
        useDeepAK8=True,
        useDeepCSV=True,
        topTagger='baseline',
        recomb_weights=None,
        iov='2016',
        bkgEst=False,
        noSyst=False,
        blinding=False,
        systematics=['nominal', 'pileup', 'pdf', 'q2', 'ttag_pt1'],
        anacats=['2t0bcen'],
        debug=False,
        produce_ntuple=False,
        ntuple_mode='accumulator',
        ntuple_output_dir=None,
        ntuple_tree_name='ttbar',
        ntuple_columns='full',
        sample_metadata=None,
    ):
        self.iov = iov
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.deepAK8Cut = deepAK8Cut
        self.useDeepAK8 = useDeepAK8
        self.useDeepCSV = useDeepCSV
        self.means_stddevs = defaultdict()
        self.bkgEst = bkgEst
        self.noSyst = noSyst
        self.systematics = systematics
        self.blinding = blinding
        self.debug = debug
        self.produce_ntuple = produce_ntuple
        self.ntuple_mode = ntuple_mode
        self.ntuple_output_dir = ntuple_output_dir
        self.ntuple_tree_name = ntuple_tree_name
        self.ntuple_columns = ntuple_columns_for_preset(ntuple_columns)
        self.store_ntuple_accumulator = produce_ntuple and ntuple_mode == 'accumulator'
        self.write_ntuple_chunks = produce_ntuple and ntuple_mode == 'chunks'
        self.sample_metadata = copy.deepcopy(sample_metadata or {})

        self.logger = Logger(mode='debug' if debug else 'info')

        self.weights = {}
        self.jet_manager = Run3JetManager(
            iov=self.iov,
            systematics=self.systematics,
            no_syst=self.noSyst,
            ak8_pt_min=self.ak8PtMin,
            ht_cut=self.htCut,
        )
        self.weight_manager = Run3WeightManager(
            iov=self.iov,
            systematics=self.systematics,
            no_syst=self.noSyst,
            deepak8_cut=self.deepAK8Cut,
        )

        self.deepAK8disc = _TAGGER_WPS[deepAK8Cut][self.iov]
        if deepAK8Cut == 'tight':
            self.deepAK8low = _TAGGER_WPS['medium'][self.iov]
        elif deepAK8Cut == 'medium':
            self.deepAK8low = _TAGGER_WPS['loose'][self.iov]
        else:
            self.deepAK8low = 0.2

        # --- GloParTv3 per-pT recombination tagger (opt-in) ---------------------
        # topTagger='recomb' replaces the baseline TopvsQCD ratio with the learned
        # per-pT logistic score. The deploy JSON (build_recomb_deploy.py) carries
        # the per-pT logistic params + per-pT WP thresholds matched to the baseline
        # per-pT QCD mis-tag, so the background per pT bin is unchanged and only the
        # signal efficiency rises. Score and thresholds are both in sigmoid space.
        self.topTagger = topTagger
        self._recomb = (topTagger == 'recomb')
        if self._recomb:
            if not recomb_weights:
                raise ValueError("topTagger='recomb' requires recomb_weights (deploy JSON path)")
            with open(recomb_weights) as f:
                dep = json.load(f)
            self._recomb_eps = float(dep.get('eps', gr.EPS))
            self._recomb_pt_edges = np.asarray(dep['pt_edges'], dtype=np.float64)
            self._recomb_params = {
                int(b): {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
                for b, p in dep['params'].items()
            }
            nb = len(self._recomb_pt_edges) - 1
            wp = dep['wp']
            if deepAK8Cut not in wp:
                raise KeyError(f"recomb deploy has no WP '{deepAK8Cut}' (have {list(wp)})")
            low_map = {'tight': 'medium', 'medium': 'loose'}  # antitag low WP
            self._recomb_disc_arr = np.array([wp[deepAK8Cut][str(b)] for b in range(nb)])
            if deepAK8Cut in low_map:
                self._recomb_low_arr = np.array([wp[low_map[deepAK8Cut]][str(b)] for b in range(nb)])
            else:  # 'loose' has no lower WP table -> accept everything below disc
                self._recomb_low_arr = np.zeros(nb)

        # tagger discriminant field name; 2024 uses a composite score (see _tscore)
        _tagger_fields = {
            '2023': 'particleNet_XttVsQCD',
        }
        self.tagger_field = _tagger_fields.get(self.iov)

        # trigger paths per IOV
        self.triggernames = {
            '2022': ['PFHT1050'],
            '2023': ['PFHT1050'],
            '2024': ['PFHT1050'],
            '2025': ['PFHT1050'],
        }

        self.anacats = anacats
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}

        self.histo_dict = build_output_histograms(
            anacats=self.anacats,
            systematics=self.systematics,
            no_syst=self.noSyst,
            produce_ntuple=self.store_ntuple_accumulator,
            produce_ntuple_chunks=self.write_ntuple_chunks,
            ntuple_columns=ntuple_columns,
        )

    def _tscore(self, jet):
        """Return the top-tagger discriminant for one or more jets.

        topTagger='recomb': learned per-pT GloParTv3 recombination score (sigmoid).
        Else 2024: mass-decorrelated GloParTv3 TopvsQCD =
              (TopbWqq + TopbWq) / (TopbWqq + TopbWq + QCD)
        Other IOVs: single NanoAOD field stored in self.tagger_field.
        Works on flat (one jet/event) or jagged (lists of jets) inputs.
        """
        if self._recomb:
            return self._recomb_tscore(jet)
        if self.iov == '2024':
            num = jet.globalParT3_TopbWqq + jet.globalParT3_TopbWq
            return num / (num + jet.globalParT3_QCD)
        return jet[self.tagger_field]

    def _recomb_tscore(self, jet):
        """Per-pT logistic recombination score (sigmoid in [0,1]) for ``jet``.

        Handles both a flat per-event jet (e.g. jet0) and a jagged collection
        (e.g. the two leading jets used for score-sorting).
        """
        pt = jet.pt
        jagged = pt.ndim > 1
        if jagged:
            counts = ak.num(pt, axis=1)
            ptn = ak.to_numpy(ak.flatten(pt))
            heads = {h: ak.to_numpy(ak.flatten(jet["globalParT3_" + h])) for h in gr.RAW_HEADS}
        else:
            ptn = ak.to_numpy(pt)
            heads = {h: ak.to_numpy(jet["globalParT3_" + h]) for h in gr.RAW_HEADS}
        X = gr.build_features_eng(heads, eps=self._recomb_eps)
        idx = gr.pt_bin_index(ptn, self._recomb_pt_edges)
        logodds = np.zeros(len(X), dtype=np.float64)
        for b, params in self._recomb_params.items():
            m = idx == b
            if np.any(m):
                logodds[m] = gr.apply_logistic(X[m], params)
        score = 1.0 / (1.0 + np.exp(-logodds))
        return ak.unflatten(score, ak.to_numpy(counts)) if jagged else ak.Array(score)

    def _disc_thr(self, jet):
        """Tag (signal) threshold — scalar for baseline, per-pT for recomb."""
        if not self._recomb:
            return self.deepAK8disc
        idx = gr.pt_bin_index(ak.to_numpy(jet.pt), self._recomb_pt_edges)
        return ak.Array(self._recomb_disc_arr[idx])

    def _low_thr(self, jet):
        """Antitag lower threshold — scalar for baseline, per-pT for recomb."""
        if not self._recomb:
            return self.deepAK8low
        idx = gr.pt_bin_index(ak.to_numpy(jet.pt), self._recomb_pt_edges)
        return ak.Array(self._recomb_low_arr[idx])

    @staticmethod
    def _nearby_jet_label(jet, ak4s, ak8s):
        """Return a per-event numpy string array describing what neighbors `jet`.

        Labels:
          'ak4_nearby'    — an AK4 exists in the annulus 0.4 < dR < 0.8 around the jet
          'ak8_nearby'    — a different AK8 exists within dR < 0.8 (excluding self via dR > 1e-6)
          'no_jet_nearby' — neither of the above
        """
        ak4_pairs = ak.cartesian({"ak8": ak.singletons(jet), "ak4": ak4s}, axis=1, nested=True)
        ak4_dr = ak4_pairs["ak8"].p4.delta_r(ak4_pairs["ak4"].p4)
        has_nearby_ak4 = ak.to_numpy(
            ak.flatten(ak.any((ak4_dr > _DR_NEARBY_INNER) & (ak4_dr < _DR_AK8), axis=2), axis=1)
        )

        ak8_pairs = ak.cartesian({"ak8": ak.singletons(jet), "fat": ak8s}, axis=1, nested=True)
        ak8_dr = ak8_pairs["ak8"].p4.delta_r(ak8_pairs["fat"].p4)
        has_nearby_ak8 = ak.to_numpy(
            ak.flatten(ak.any((ak8_dr < _DR_AK8) & (ak8_dr > 1e-6), axis=2), axis=1)
        )

        return np.where(
            has_nearby_ak8,
            "ak8_nearby",
            np.where(has_nearby_ak4, "ak4_nearby", "no_jet_nearby"),
        )

    def _fill_ntuple(
        self, output, correction, events, labels_and_categories,
        jetpt, jeteta, jetphi, jetmsd, tdisc_s0,
        jetpt1, jeteta1, jetphi1, jetmsd1, tdisc_s1,
        ttbarmass, ht, rapidity, chi, jety, jety1, evtweights,
    ):
        """Accumulate flat ntuple columns for all selected events."""
        branches = self._build_ntuple_branches(
            events, labels_and_categories,
            jetpt, jeteta, jetphi, jetmsd, tdisc_s0,
            jetpt1, jeteta1, jetphi1, jetmsd1, tdisc_s1,
            ttbarmass, ht, rapidity, chi, jety, jety1, evtweights,
        )

        for name, values in branches.items():
            output["ntuple"][name] += processor.column_accumulator(values)

    def _build_ntuple_branches(
        self, events, labels_and_categories,
        jetpt, jeteta, jetphi, jetmsd, tdisc_s0,
        jetpt1, jeteta1, jetphi1, jetmsd1, tdisc_s1,
        ttbarmass, ht, rapidity, chi, jety, jety1, evtweights,
    ):
        """Build flat ntuple arrays for the selected nominal chunk."""
        anacat_arr = np.full(len(events), -1, dtype=np.int64)
        for _i, (_lbl, _mask) in enumerate(labels_and_categories.items()):
            anacat_arr[ak.to_numpy(_mask)] = _i

        def _col(arr, dtype=np.float32):
            return ak.to_numpy(arr).astype(dtype)

        branches = {
            "jet0_pt":       _col(jetpt),
            "jet0_eta":      _col(jeteta),
            "jet0_phi":      _col(jetphi),
            "jet0_msd":      _col(jetmsd),
            "jet0_tdisc":    _col(tdisc_s0),
            "jet1_pt":       _col(jetpt1),
            "jet1_eta":      _col(jeteta1),
            "jet1_phi":      _col(jetphi1),
            "jet1_msd":      _col(jetmsd1),
            "jet1_tdisc":    _col(tdisc_s1),
            "ttbarmass":     _col(ttbarmass),
            "ht":            _col(ht),
            "dy":            _col(rapidity),
            "chi":           _col(chi),
            "jet0_rapidity": _col(jety),
            "jet1_rapidity": _col(jety1),
            "weight":        _col(evtweights),
            "anacat":        _col(anacat_arr, dtype=np.int64),
            "run":           _col(events.run, dtype=np.int64),
            "lumi":          _col(events.luminosityBlock, dtype=np.int64),
            "event":         _col(events.event, dtype=np.int64),
        }
        return {name: branches[name] for name in self.ntuple_columns}

    def _write_ntuple_chunk(self, events, labels_and_categories, *branch_args):
        """Write this nominal chunk to a small ROOT file and return its path."""
        if not self.ntuple_output_dir:
            raise RuntimeError("ntuple_mode='chunks' requires ntuple_output_dir")

        branches = self._build_ntuple_branches(events, labels_and_categories, *branch_args)
        n_rows = len(next(iter(branches.values()))) if branches else 0
        if n_rows == 0:
            return None

        import uproot

        metadata = events.metadata
        dataset = str(metadata.get('dataset', self.sample_metadata.get('sample', 'dataset')))
        filename = str(metadata.get('filename', metadata.get('file', 'unknown_file')))
        entrystart = metadata.get('entrystart', metadata.get('entry_start', 'start'))
        entrystop = metadata.get('entrystop', metadata.get('entry_stop', 'stop'))
        chunk_key = f"{dataset}|{filename}|{entrystart}|{entrystop}"
        chunk_hash = hashlib.sha1(chunk_key.encode()).hexdigest()[:16]
        safe_dataset = ''.join(c if c.isalnum() or c in '._-' else '_' for c in dataset)

        chunk_name = f"{safe_dataset}_{chunk_hash}_{entrystart}_{entrystop}.root"
        chunk_path = os.path.join(self.ntuple_output_dir, chunk_name)
        local_tmp_dir = None

        if _is_xrootd_path(chunk_path):
            local_tmp_dir = tempfile.mkdtemp(prefix="ttbar_ntuple_chunk_")
            tmp_path = os.path.join(local_tmp_dir, f"{chunk_name}.tmp.{os.getpid()}")
        else:
            os.makedirs(self.ntuple_output_dir, exist_ok=True)
            tmp_path = f"{chunk_path}.tmp.{os.getpid()}"

        with uproot.recreate(tmp_path) as fout:
            fout.mktree(self.ntuple_tree_name, {name: values.dtype for name, values in branches.items()})
            fout[self.ntuple_tree_name].extend(branches)

        if _is_xrootd_path(chunk_path):
            try:
                _copy_to_xrootd(tmp_path, chunk_path)
            finally:
                shutil.rmtree(local_tmp_dir, ignore_errors=True)
        else:
            os.replace(tmp_path, chunk_path)
            if not os.path.exists(chunk_path):
                raise RuntimeError(f"ntuple chunk write did not leave a visible file: {chunk_path}")
        return chunk_path

    def _fill_kinematic_hists(
        self, output, systematic, i, icat, weights,
        jetmsd, jetmsd1, ttbarmass, rapidity, chi, ht,
        jetpt, jeteta, jetphi, jety,
        jetpt1, jeteta1, jetphi1, jety1,
    ):
        """Fill all kinematic histograms for one analysis category and one systematic."""
        w  = weights[icat]
        kw = dict(systematic=systematic, anacat=i)
        output['jetmsd'].fill(      **kw, jetmsd=jetmsd[icat],         weight=w)
        output['ttbarmass'].fill(   **kw, ttbarmass=ttbarmass[icat],   weight=w)
        output['mtt_vs_mt'].fill(   **kw, jetmass=jetmsd[icat], ttbarmass=ttbarmass[icat], weight=w)
        output['jetmsd1'].fill(     **kw, jetmsd=jetmsd1[icat],        weight=w)
        output['jetdy'].fill(       **kw, jetdy=rapidity[icat],         weight=w)
        output['chi'].fill(         **kw, chi=chi[icat],                weight=w)
        output['ht'].fill(          **kw, ht=ht[icat],                  weight=w)
        output['jet0_pt'].fill(     **kw, jetpt=jetpt[icat],            weight=w)
        output['jet0_eta'].fill(    **kw, jeteta=jeteta[icat],          weight=w)
        output['jet0_phi'].fill(    **kw, jetphi=jetphi[icat],          weight=w)
        output['jet0_rapidity'].fill(**kw, jety=jety[icat],             weight=w)
        output['jet1_pt'].fill(     **kw, jetpt=jetpt1[icat],           weight=w)
        output['jet1_eta'].fill(    **kw, jeteta=jeteta1[icat],         weight=w)
        output['jet1_phi'].fill(    **kw, jetphi=jetphi1[icat],         weight=w)
        output['jet1_rapidity'].fill(**kw, jety=jety1[icat],            weight=w)

    def _build_normalization_metadata(self, sumw_raw, sumw2_raw, scale_factor, applied, reason=None):
        sample_metadata = copy.deepcopy(self.sample_metadata)
        year = str(sample_metadata.get('year', self.iov))
        lumi_pb = _LUMI_PB.get(year)

        normalization = {
            'applied': bool(applied),
            'sample': sample_metadata.get('sample'),
            'subsample': sample_metadata.get('subsample'),
            'year': year,
            'is_mc': bool(sample_metadata.get('is_mc', False)),
            'xsec_pb': sample_metadata.get('xsec_pb'),
            'lumi_pb': lumi_pb,
            'sumw_raw': float(sumw_raw),
            'sumw2_raw': float(sumw2_raw),
            'scale_factor': float(scale_factor),
        }
        if reason is not None:
            normalization['reason'] = reason

        return sample_metadata, normalization

    @staticmethod
    def _build_scaled_cutflow(cutflow, scale_factor):
        scaled_cutflow = {}
        for key, value in cutflow.items():
            if key == 'sumw2':
                scaled_cutflow[key] = float(value) * (scale_factor ** 2)
            else:
                scaled_cutflow[key] = float(value) * scale_factor
        return scaled_cutflow

    @staticmethod
    def _scale_mapping(values, scale_factor, power=1):
        return {
            key: float(value) * (scale_factor ** power)
            for key, value in values.items()
        }

    @staticmethod
    def _sum_weight_array(weights, mask=None, square=False):
        values = ak.to_numpy(weights)
        if mask is not None:
            values = values[ak.to_numpy(mask)]
        if square:
            values = values ** 2
        return float(np.sum(values))

    def _fill_cutflow_table_step(self, output, key, mask=None, weights=None, count=None):
        """Fill the structured cutflow table maps without changing selections."""
        if count is None:
            count = int(ak.sum(mask)) if mask is not None else len(weights)

        output['cutflow_unweighted'][key] += float(count)
        if weights is None:
            return

        output['cutflow_weighted'][key] += self._sum_weight_array(weights, mask=mask)
        output['cutflow_weighted2'][key] += self._sum_weight_array(weights, mask=mask, square=True)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        logger.debug(
            'memory:%s:start %s preprocessor: %s',
            time.time(), events.metadata['dataset'], get_memory_usage(),
        )

        dataset = events.metadata['dataset']
        nEvents = len(events.event)
        self.logger.debug('preprocessor input: dataset=%s, events=%d', dataset, nEvents)

        # remove QCD events with large gen weights
        if "QCD" in dataset:
            if dataset not in self.means_stddevs:
                average = np.average(events.genWeight)
                stddev  = np.std(events.genWeight)
                self.means_stddevs[dataset] = (average, stddev)
            average, stddev = self.means_stddevs[dataset]
            self.logger.debug(
                'QCD genWeight stats: before=%d, average=%s, stddev=%s',
                len(events), average, stddev,
            )
            if stddev == 0 or not np.isfinite(stddev):
                self.logger.debug(
                    'QCD genWeight filter skipped: stddev=%s, keeping all %d events',
                    stddev, len(events),
                )
            else:
                vals = (events.genWeight - average) / stddev
                genweight_mask = np.abs(vals) < 2
                n_pass_genweight = int(ak.sum(genweight_mask))
                self.logger.debug(
                    'QCD genWeight filter: before=%d, pass=%d, fail=%d',
                    len(events), n_pass_genweight, len(events) - n_pass_genweight,
                )
                events = events[genweight_mask]
            self.logger.debug(
                'QCD preprocessor output: raw=%d, kept=%d',
                nEvents, len(events),
            )

        isData = ('data' in dataset) or ('SingleMu' in dataset)
        corrections = self.jet_manager.build_corrections(events, isData)

        if corrections is None:
            return self.process_analysis(events, 'nominal', nEvents)

        outputs = []
        for collections, name in corrections:
            outputs.append(self.process_analysis(update(events, collections), name, nEvents))

        return outputs[0]

    def process_analysis(self, events, correction, nEvents):
        dataset  = events.metadata['dataset']
        self.logger.debug('start processor: correction=%s, memory=%s', correction, get_memory_usage())

        isNominal = (correction == 'nominal')
        isData    = ('data' in dataset) or ('SingleMu' in dataset)

        output = self.histo_dict

        if isNominal:
            output['cutflow']['all events 1'] += nEvents
            self._fill_cutflow_table_step(output, 'input_events', count=nEvents)
            self.logger.debug(
                'cutflow all events 1 filled: original_chunk_events=%d, '
                'events_entering_analysis=%d',
                nEvents, len(events),
            )

        # --- lumi mask ---
        if isData:
            lumi_mask = np.array(getLumiMask(self.iov)(events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
            if isNominal:
                output['cutflow']['after_lumimask'] += len(events)
                print(f"[CUTFLOW] after lumimask: {len(events)}")

        # --- blinding (data only, keep every 10th event) ---
        if self.blinding and isData:
            events = events[::10]
            self.logger.debug('after blinding: events=%d', len(events))

        # --- trigger ---
        selection = PackedSelection()
        trig_paths = self.triggernames[self.iov]

        if 'HLT' not in events.fields:
            warnings.warn(
                f"HLT branch missing for IOV {self.iov}; accepting all events."
            )
            selection.add('trigger', np.ones(len(events), dtype=bool))
        else:
            available = [p for p in trig_paths if p in events.HLT.fields]
            if not available:
                warnings.warn(
                    f"None of {trig_paths} present in HLT for IOV {self.iov}; "
                    "accepting all events."
                )
                selection.add('trigger', np.ones(len(events), dtype=bool))
            else:
                mask = events.HLT[available[0]]
                for p in available[1:]:
                    mask = mask | events.HLT[p]
                selection.add('trigger', mask)
                self.logger.debug(
                    'trigger paths found for %s: %s, pass=%d/%d',
                    self.iov, available, int(ak.sum(mask)), len(events),
                )

        # --- build jet collections ---
        FatJets, SubJets, Jets, GenJets, GenJetAK8, SubGenJetAK8 = (
            self.jet_manager.prepare_analysis_objects(events, isData)
        )
        self.logger.debug(
            'prepared objects: events=%d, total FatJet=%d, total Jet=%d, '
            'events_with>=2FatJet=%d',
            len(events), int(ak.sum(ak.num(FatJets))), int(ak.sum(ak.num(Jets))),
            int(ak.sum(ak.num(FatJets) >= 2)),
        )
        run  = events.run.to_numpy()
        lumi = events.luminosityBlock.to_numpy()
        evt  = events.event.to_numpy()

        if 'globalParT3_TopbWqq' not in events.FatJet.fields:
            print("\n--- FatJet variables (globalParT3_TopbWqq missing) ---")
            for var in events.FatJet.fields:
                print("FatJet_" + var)
            raise RuntimeError(
                f"globalParT3_TopbWqq not found in FatJet fields for dataset {dataset}"
            )
        logger.debug('memory:%s: get nanoAOD objects %s:%s', time.time(), correction, get_memory_usage())

        if len(events) < 10:
            self.logger.debug(
                'early return before weights/baseline cutflow: events=%d < 10 '
                '(original_chunk_events=%d, correction=%s)',
                len(events), nEvents, correction,
            )
            return output

        # --- event weights ---
        if isData:
            evtweights = np.ones(len(events))
        else:
            if "LHEWeight_originalXWGTUP" not in events.fields:
                evtweights = events.genWeight
            else:
                evtweights = events.LHEWeight_originalXWGTUP

        if isNominal:
            output['cutflow']['all events'] += len(FatJets)
            output['cutflow']['sumw']        += np.sum(evtweights)
            output['cutflow']['sumw2']       += np.sum(evtweights ** 2)
            self._fill_cutflow_table_step(
                output, 'analysis_events', count=len(events), weights=evtweights,
            )
            self.logger.debug(
                'cutflow all events filled: events=%d, sumw=%s, sumw2=%s',
                len(FatJets), np.sum(evtweights), np.sum(evtweights ** 2),
            )

        # --- baseline jet selection ---
        FatJets, jet_masks = self.jet_manager.baseline_masks(events, FatJets, Jets)
        selection.add('htCut',      jet_masks['htCut'])
        selection.add('metfilter',  getMETFilter(self.iov, events))
        selection.add('jetkincut',  jet_masks['jetkincut'])
        selection.add('twoFatJets', jet_masks['twoFatJets'])

        if isNominal:
            cuts = []
            for cut in selection.names:
                cuts.append(cut)
                cumulative_mask = selection.all(*cuts)
                n = int(ak.sum(cumulative_mask))
                output['cutflow'][cut] += n
                self._fill_cutflow_table_step(
                    output, cut, mask=cumulative_mask, weights=evtweights,
                )
                print(f"[CUTFLOW] after {cut} (cumulative): {n}")

        eventCut = selection.all(*selection.names)
        self.logger.debug(
            'combined preselection mask: pass=%d/%d, cuts=%s',
            int(ak.sum(eventCut)), len(events), selection.names,
        )

        FatJets    = FatJets[eventCut]
        SubJets    = SubJets[eventCut]
        Jets       = Jets[eventCut]
        evtweights = evtweights[eventCut]
        events     = events[eventCut]
        run        = run[eventCut]
        lumi       = lumi[eventCut]
        evt        = evt[eventCut]

        if isNominal:
            output['cutflow']['after_eventCut'] += len(events)
            self._fill_cutflow_table_step(
                output, 'preselection', count=len(events), weights=evtweights,
            )
            print(f"[CUTFLOW] after all preselection (eventCut): {len(events)}")
        logger.debug(f"Length of event {len(events)}")
        if len(events) < 10:
            self.logger.debug(
                'early return after baseline eventCut: events=%d < 10 '
                '(original_chunk_events=%d, correction=%s)',
                len(events), nEvents, correction,
            )
            return output

        if not isData:
            GenJets = GenJets[eventCut]
            if GenJetAK8 is not None:
                GenJetAK8 = GenJetAK8[eventCut]
            if SubGenJetAK8 is not None:
                SubGenJetAK8 = SubGenJetAK8[eventCut]

        # --- pre-selection truth study ---
        if isNominal and not isData:
            truth_counts = truthstudy_counts(
                genparts=events.GenPart,
                fatjets=FatJets,
                subJets=SubJets,
                jets=Jets,
                dr_ak8=_DR_AK8,
                dr_ak4=_DR_AK4,
            )
            output["truthstudy"]["n_hadtop"]         += truth_counts["n_hadtop"]
            output["truthstudy"]["n_hadtop_ak8"]     += truth_counts["n_hadtop_ak8"]
            output["truthstudy"]["n_hadtop_ak8_ak4"] += truth_counts["n_hadtop_ak8_ak4"]

        # --- sort FatJets by pT, then assign jet0 as the higher-scoring one ---
        pt_order    = ak.argsort(FatJets.pt, ascending=False)
        sorted_jets = FatJets[pt_order]
        top_two_jets = sorted_jets[:, :2]
        score_order = ak.argsort(self._tscore(top_two_jets), ascending=False)
        score_sorted_jets = top_two_jets[score_order]
        jet0 = score_sorted_jets[:, 0]
        jet1 = score_sorted_jets[:, 1]

        mcut_s0 = (self.minMSD < jet0.msoftdrop) & (jet0.msoftdrop < self.maxMSD)
        mcut_s1 = (self.minMSD < jet1.msoftdrop) & (jet1.msoftdrop < self.maxMSD)

        # signal region: both jets pass the tagger (per-pT thresholds in recomb mode)
        ttag_s0 = self._tscore(jet0) > self._disc_thr(jet0)
        ttag_s1 = (self._tscore(jet1) > self._disc_thr(jet1)) & mcut_s1

        # antitag (fail) region: leading passes, subleading in the fail-but-above-low window
        antitag_disc = (
            (self._tscore(jet1) < self._disc_thr(jet1))
            & (self._tscore(jet1) > self._low_thr(jet1))
        )
        antitag = antitag_disc & ttag_s0 & mcut_s1

        # back-to-back topology + subjet requirements
        dPhiCut     = np.abs(jet0.p4.delta_phi(jet1.p4)) > _DPHI_CUT
        hasSubjets0 = (jet0.subJetIdx1 > -1) & (jet0.subJetIdx2 > -1)
        hasSubjets1 = (jet1.subJetIdx1 > -1) & (jet1.subJetIdx2 > -1)
        GoodSubjets = hasSubjets0 & hasSubjets1
        ttbarcandCuts = dPhiCut & GoodSubjets

        if isNominal:
            self._fill_cutflow_table_step(
                output, 'dphi', mask=dPhiCut, weights=evtweights,
            )
            self._fill_cutflow_table_step(
                output, 'ttbarcand', mask=ttbarcandCuts, weights=evtweights,
            )

        # save copies before slicing — run/lumi/evt are masked by the signal requirement
        ttag_s0_precut = ttag_s0
        ttag_s1_precut = ttag_s1

        antitag    = antitag[ttbarcandCuts]
        ttag_s0    = ttag_s0[ttbarcandCuts]
        ttag_s1    = ttag_s1[ttbarcandCuts]
        jet0       = jet0[ttbarcandCuts]
        jet1       = jet1[ttbarcandCuts]
        FatJets    = FatJets[ttbarcandCuts]
        Jets       = Jets[ttbarcandCuts]
        SubJets    = SubJets[ttbarcandCuts]
        events     = events[ttbarcandCuts]
        evtweights = evtweights[ttbarcandCuts]
        run  = run[ttbarcandCuts & ttag_s0_precut & ttag_s1_precut]
        lumi = lumi[ttbarcandCuts & ttag_s0_precut & ttag_s1_precut]
        evt  = evt[ttbarcandCuts & ttag_s0_precut & ttag_s1_precut]

        if isNominal:
            output['cutflow']['after_ttbarcandCuts'] += len(events)
            n_dPhi    = int(ak.sum(dPhiCut))
            n_subjets = int(ak.sum(GoodSubjets))
            n_both    = int(ak.sum(dPhiCut & GoodSubjets))
            print(f"[CUTFLOW] after ttbarcandCuts: {len(events)}  "
                  f"(dPhiCut alone: {n_dPhi}, GoodSubjets alone: {n_subjets}, both: {n_both})")

        if isNominal:
            before = len(output["event_list"]["run"])
            output["event_list"]["run"]   += list(run)
            output["event_list"]["lumi"]  += list(lumi)
            output["event_list"]["event"] += list(evt)
            if self.debug:
                after = len(output["event_list"]["run"])
                print(after - before, len(run), "   if different then is wrong")
                df = pd.DataFrame(output["event_list"])
                print("rows:", len(df))
                print("duplicates:", df.duplicated(["run", "lumi", "event"]).sum())

        if not isData:
            GenJets = GenJets[ttbarcandCuts]
            if GenJetAK8 is not None:
                GenJetAK8 = GenJetAK8[ttbarcandCuts]
            if SubGenJetAK8 is not None:
                SubGenJetAK8 = SubGenJetAK8[ttbarcandCuts]

        logger.debug('memory:%s: apply event cuts %s:%s', time.time(), correction, get_memory_usage())

        # --- derived kinematic quantities ---
        third_jet_mask = ak.num(FatJets) > 2
        jet2         = FatJets[third_jet_mask][:, 2]
        dR_jet0_jet2 = jet0[third_jet_mask].p4.delta_r(jet2.p4)
        dR_jet1_jet2 = jet1[third_jet_mask].p4.delta_r(jet2.p4)

        ttbarmass = (jet0.p4 + jet1.p4).mass
        ht = ak.sum(Jets[(Jets.pt > _AK4_PT_MIN) & (np.abs(Jets.eta) < _AK4_ETA_MAX)].pt, axis=1)

        tdisc_s0 = self._tscore(jet0)
        tdisc_s1 = self._tscore(jet1)

        rapidity = getRapidity(jet0.p4) - getRapidity(jet1.p4)
        chi      = np.exp(np.abs(rapidity))

        jetpt  = jet0.p4.pt
        jeteta = jet0.p4.eta
        jety   = getRapidity(jet0.p4)
        jetphi = jet0.p4.phi
        jetmsd = jet0.msoftdrop

        jetpt1  = jet1.p4.pt
        jeteta1 = jet1.p4.eta
        jety1   = getRapidity(jet1.p4)
        jetphi1 = jet1.p4.phi
        jetmsd1 = jet1.msoftdrop

        jet0_abs_eta = np.abs(jet0.eta)
        jet1_abs_eta = np.abs(jet1.eta)

        # per-event label describing what jet objects neighbor each leading FatJet
        jet0_nearby_label = self._nearby_jet_label(jet0, Jets, FatJets)
        jet1_nearby_label = self._nearby_jet_label(jet1, Jets, FatJets)

        # --- gen-level matching (nominal MC only) ---
        gen_top_match_info  = None
        genjetak8_match_info = None
        if isNominal and not isData:
            gen_top_match_info = build_gen_top_match_info(
                genparts=events.GenPart, jet0=jet0, jet1=jet1, dr_match=_DR_AK8,
            )
            genjetak8_match_info = build_top_aligned_genjetak8_match_info(
                genparts=events.GenPart,
                genjetak8=GenJetAK8,
                subgenjetak8=SubGenJetAK8,
                jet0=jet0,
                jet1=jet1,
                dr_match=_DR_AK8,
                top_align_dr=_DR_AK8,
            )
            output["truthstudy"]["n_selected_allhad"]       += int(ak.sum(gen_top_match_info["event_mask"]))
            output["truthstudy"]["n_selected_jet0_matched"] += int(ak.sum(gen_top_match_info["jet0_is_matched"]))
            output["truthstudy"]["n_selected_jet1_matched"] += int(ak.sum(gen_top_match_info["jet1_is_matched"]))
            output["truthstudy"]["n_selected_both_matched"] += int(ak.sum(gen_top_match_info["both_jets_matched"]))
            if genjetak8_match_info is not None:
                output["truthstudy"]["n_selected_jet0_genak8_matched"] += int(ak.sum(genjetak8_match_info["jet0_is_matched"]))
                output["truthstudy"]["n_selected_jet1_genak8_matched"] += int(ak.sum(genjetak8_match_info["jet1_is_matched"]))
                output["truthstudy"]["n_selected_both_genak8_matched"] += int(ak.sum(genjetak8_match_info["both_jets_matched"]))

        # --- analysis categories and weights ---
        labels_and_categories = build_analysis_categories(
            antitag=antitag,
            ttag_s0=ttag_s0,
            ttag_s1=ttag_s1,
            rapidity=rapidity,
            anacats=self.anacats,
        )
        if isNominal:
            self._fill_cutflow_table_step(
                output, 'tag_jet0', mask=ttag_s0, weights=evtweights,
            )
            self._fill_cutflow_table_step(
                output, 'tag_2tag', mask=(ttag_s0 & ttag_s1), weights=evtweights,
            )
            self._fill_cutflow_table_step(
                output, 'antitag', mask=antitag, weights=evtweights,
            )
            print(f"[CUTFLOW] antitag: {int(ak.sum(antitag))}, ttag_s0: {int(ak.sum(ttag_s0))}, "
                  f"ttag_s1: {int(ak.sum(ttag_s1))}, 2tag: {int(ak.sum(ttag_s0 & ttag_s1))}")
            for lbl, cat in labels_and_categories.items():
                self._fill_cutflow_table_step(
                    output, f'category_{lbl}', mask=cat, weights=evtweights,
                )
                print(f"[CUTFLOW] category '{lbl}': {int(ak.sum(cat))}")

        self.weights[correction] = self.weight_manager.build_weights(
            dataset=dataset,
            events=events,
            evtweights=evtweights,
            is_data=isData,
            jet0=jet0,
            jet1=jet1,
            ttag2=(ttag_s0 & ttag_s1),
            antitag=antitag,
        )

        # --- flat ntuple output ---
        if isNominal and self.store_ntuple_accumulator:
            self._fill_ntuple(
                output, correction, events, labels_and_categories,
                jetpt, jeteta, jetphi, jetmsd, tdisc_s0,
                jetpt1, jeteta1, jetphi1, jetmsd1, tdisc_s1,
                ttbarmass, ht, rapidity, chi, jety, jety1, evtweights,
            )
        elif isNominal and self.write_ntuple_chunks:
            chunk_path = self._write_ntuple_chunk(
                events,
                labels_and_categories,
                jetpt, jeteta, jetphi, jetmsd, tdisc_s0,
                jetpt1, jeteta1, jetphi1, jetmsd1, tdisc_s1,
                ttbarmass, ht, rapidity, chi, jety, jety1, evtweights,
            )
            if chunk_path:
                output["ntuple_chunks"] += [chunk_path]

        # --- per-category histogram filling ---
        for i, (ilabel, icat) in enumerate(labels_and_categories.items()):
            if isNominal:
                output['cutflow'][ilabel] += len(events.event[icat])

                dR_min_jet2 = ak.where(dR_jet0_jet2 < dR_jet1_jet2, dR_jet0_jet2, dR_jet1_jet2)
                output['dR_min_jet2'].fill(
                    systematic=correction,
                    dr=dR_min_jet2,
                    ttbarmass=ttbarmass[third_jet_mask],
                    anacat=i,
                    weight=self.weights[correction].weight()[third_jet_mask],
                )

            self._fill_kinematic_hists(
                output, correction, i, icat,
                self.weights[correction].weight(),
                jetmsd, jetmsd1, ttbarmass, rapidity, chi, ht,
                jetpt, jeteta, jetphi, jety,
                jetpt1, jeteta1, jetphi1, jety1,
            )
            if correction == "nominal":
                output['mtt_vs_dy_vs_chi'].fill(
                    ttbarmass=ttbarmass[icat],
                    jetdy=np.abs(rapidity[icat]),
                    chi=chi[icat],
                    weight=self.weights[correction].weight()[icat],
                )

            # gen truth histograms
            if gen_top_match_info is not None:
                truth_cat_mask      = icat[gen_top_match_info["event_mask"]]
                truth_event_weights = self.weights[correction].weight()[gen_top_match_info["event_mask"]]
                truth_weights       = truth_event_weights[truth_cat_mask]

                if ak.sum(truth_cat_mask) == 0:
                    continue

                output["gen_mt"].fill(
                    systematic=correction, anacat=i,
                    gentopmass=gen_top_match_info["gen_top0"].mass[truth_cat_mask],
                    weight=truth_weights,
                )
                output["gen_mt"].fill(
                    systematic=correction, anacat=i,
                    gentopmass=gen_top_match_info["gen_top1"].mass[truth_cat_mask],
                    weight=truth_weights,
                )
                output["gen_mttbar"].fill(
                    systematic=correction, anacat=i,
                    ttbarmass=gen_top_match_info["top_pair_mass"][truth_cat_mask],
                    weight=truth_weights,
                )
                output["jet0_gen_dr"].fill(
                    systematic=correction, anacat=i,
                    dr=gen_top_match_info["jet0_dr"][truth_cat_mask],
                    weight=truth_weights,
                )
                output["jet1_gen_dr"].fill(
                    systematic=correction, anacat=i,
                    dr=gen_top_match_info["jet1_dr"][truth_cat_mask],
                    weight=truth_weights,
                )

            if genjetak8_match_info is not None:
                genak8_truth_cat_mask = icat[genjetak8_match_info["event_mask"]]
                genak8_event_weights  = self.weights[correction].weight()[genjetak8_match_info["event_mask"]]
                genak8_truth_weights  = genak8_event_weights[genak8_truth_cat_mask]

                output["gen_jetmsd_reco_jetmsd"].fill(
                    systematic=correction, anacat=i,
                    abs_eta=jet0_abs_eta[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    jet_nearby=jet0_nearby_label[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    genjetmass=genjetak8_match_info["jet0_genjet"].mass[genak8_truth_cat_mask],
                    jetmsd=jetmsd[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    weight=genak8_truth_weights,
                )
                output["gen_jetmsd_reco_jetmsd"].fill(
                    systematic=correction, anacat=i,
                    abs_eta=jet1_abs_eta[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    jet_nearby=jet1_nearby_label[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    genjetmass=genjetak8_match_info["jet1_genjet"].mass[genak8_truth_cat_mask],
                    jetmsd=jetmsd1[genjetak8_match_info["event_mask"]][genak8_truth_cat_mask],
                    weight=genak8_truth_weights,
                )

                jet0_genak8_massres = (
                    jetmsd[genjetak8_match_info["event_mask"]] - genjetak8_match_info["jet0_genjet"].mass
                ) / genjetak8_match_info["jet0_genjet"].mass
                jet1_genak8_massres = (
                    jetmsd1[genjetak8_match_info["event_mask"]] - genjetak8_match_info["jet1_genjet"].mass
                ) / genjetak8_match_info["jet1_genjet"].mass
                jet0_genak8_truth_cat_mask = genak8_truth_cat_mask & genjetak8_match_info["jet0_is_matched"]
                jet1_genak8_truth_cat_mask = genak8_truth_cat_mask & genjetak8_match_info["jet1_is_matched"]

                output["jet_mass_resolution"].fill(
                    systematic=correction, anacat=i,
                    abs_eta=jet0_abs_eta[genjetak8_match_info["event_mask"]][jet0_genak8_truth_cat_mask],
                    jet_nearby=jet0_nearby_label[genjetak8_match_info["event_mask"]][jet0_genak8_truth_cat_mask],
                    massres=jet0_genak8_massres[jet0_genak8_truth_cat_mask],
                    weight=genak8_event_weights[jet0_genak8_truth_cat_mask],
                )
                output["jet_mass_resolution"].fill(
                    systematic=correction, anacat=i,
                    abs_eta=jet1_abs_eta[genjetak8_match_info["event_mask"]][jet1_genak8_truth_cat_mask],
                    jet_nearby=jet1_nearby_label[genjetak8_match_info["event_mask"]][jet1_genak8_truth_cat_mask],
                    massres=jet1_genak8_massres[jet1_genak8_truth_cat_mask],
                    weight=genak8_event_weights[jet1_genak8_truth_cat_mask],
                )

            output['weights'][correction]     += np.sum(self.weights[correction].weight())
            output['systematics'][correction] += len(events.event[icat])

            if isNominal:
                for syst in self.weights[correction].variations:
                    self._fill_kinematic_hists(
                        output, syst, i, icat,
                        self.weights[correction].weight(syst),
                        jetmsd, jetmsd1, ttbarmass, rapidity, chi, ht,
                        jetpt, jeteta, jetphi, jety,
                        jetpt1, jeteta1, jetphi1, jety1,
                    )

        logger.debug('memory:%s: fill histograms %s:%s', time.time(), correction, get_memory_usage())
        del self.weights[correction]

        return output

    def postprocess(self, accumulator):
        logger.debug('memory:%s: finish processor:%s', time.time(), get_memory_usage())
        sample_metadata = copy.deepcopy(self.sample_metadata)
        cutflow = accumulator.get('cutflow', {})
        sumw_raw = float(cutflow.get('sumw', 0.0))
        sumw2_raw = float(cutflow.get('sumw2', 0.0))

        scale_factor = 1.0
        applied = False
        reason = None

        xsec_pb = sample_metadata.get('xsec_pb')
        is_mc = bool(sample_metadata.get('is_mc', False))
        year = str(sample_metadata.get('year', self.iov))
        lumi_pb = _LUMI_PB.get(year)

        if not sample_metadata:
            reason = 'missing_sample_metadata'
        elif not is_mc:
            reason = 'data_sample'
        elif xsec_pb is None:
            reason = 'missing_xsec_pb'
        elif lumi_pb is None:
            reason = 'missing_lumi_pb'
        elif sumw_raw == 0.0:
            reason = 'zero_sumw'
        else:
            scale_factor = lumi_pb * float(xsec_pb) / sumw_raw
            applied = True

            for key, value in list(accumulator.items()):
                if isinstance(value, hist.Hist):
                    accumulator[key] = value * scale_factor

            if 'ntuple' in accumulator and 'weight' in accumulator['ntuple']:
                scaled_weight = (accumulator['ntuple']['weight'].value * scale_factor).astype(np.float32)
                accumulator['ntuple']['weight'] = processor.column_accumulator(scaled_weight)

        sample_metadata, normalization = self._build_normalization_metadata(
            sumw_raw=sumw_raw,
            sumw2_raw=sumw2_raw,
            scale_factor=scale_factor,
            applied=applied,
            reason=reason,
        )
        accumulator['sample_metadata'] = sample_metadata
        accumulator['normalization'] = normalization
        accumulator['cutflow_scaled'] = self._build_scaled_cutflow(cutflow, scale_factor)
        accumulator['cutflow_weighted_scaled'] = self._scale_mapping(
            accumulator.get('cutflow_weighted', {}), scale_factor,
        )
        accumulator['cutflow_weighted2_scaled'] = self._scale_mapping(
            accumulator.get('cutflow_weighted2', {}), scale_factor, power=2,
        )
        accumulator['cutflow_table_steps'] = cutflow_step_metadata(self.anacats)
        return accumulator
