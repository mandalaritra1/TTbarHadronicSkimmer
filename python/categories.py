import itertools
import numpy as np


def leptonic_top_fraction(jet):
    """GloParTv3 f_lep = L / (H + L) per jet: the leptonic share of the jet's top-likeness.

    H = TopbWqq + TopbWq (hadronic top), L = TopbWev + TopbWmv + TopbWtauhv (leptonic
    top). The tagger score H / (H + QCD) ignores L, so leptonic tops (lepton and b inside
    the AK8 jet) pass it; f_lep < c removes them (option A, leptonic-top study 2026-10-02).
    0 where H + L = 0.
    """
    H = jet.globalParT3_TopbWqq + jet.globalParT3_TopbWq
    L = jet.globalParT3_TopbWev + jet.globalParT3_TopbWmv + jet.globalParT3_TopbWtauhv
    return np.where((H + L) > 0, L / np.where((H + L) > 0, H + L, 1.0), 0.0)


def build_analysis_categories(antitag, ttag_s0, ttag_s1, rapidity, anacats):
    """Build category masks used by the processor from top-tag and rapidity regions."""

    ttag2 = (ttag_s0 & ttag_s1)
    cen = (np.abs(rapidity) < 1.0)
    fwd = (~cen)

    regs = {"cen": cen, "fwd": fwd}
    ttags = {
        "at": antitag,  # 2Dalphabet fail region
        "2t": ttag2,    # 2Dalphabet pass region
    }

    categories = {
        t[0] + y[0]: (t[1] & y[1])
        for t, y in itertools.product(ttags.items(), regs.items())
    }
    return {label: categories[label] for label in anacats}
