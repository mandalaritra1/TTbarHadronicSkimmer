import awkward as ak
import numpy as np


def _has_flag(status_flags, bit):
    return (status_flags & (1 << bit)) != 0


def get_hadronic_tops(genparts, verbose=False):
    """Return generator-level top quarks that decay hadronically.

    Selects the last-copy top (statusFlags bit 13) from each event, then
    traces through the W → last-copy W → quark daughters chain to determine
    whether each top decays hadronically (≥2 light quarks from the W).

    Parameters
    ----------
    genparts : awkward array
        NanoAOD GenPart collection (must contain pdgId, statusFlags, genPartIdxMother).
    verbose : bool
        If True, print all-had / semi-lep / di-lep event counts.

    Returns
    -------
    awkward array
        Jagged array of hadronic top GenPart records, shape [events, n_hadtop].
    """
    # bit 13 = isLastCopy in NanoAOD statusFlags
    last_copy = _has_flag(genparts.statusFlags, 13)
    is_top = (abs(genparts.pdgId) == 6) & last_copy

    # local indices per event — avoids a Python-level loop over events
    idx = ak.local_index(genparts, axis=1)
    top_idx = idx[is_top]

    if ak.all(ak.num(top_idx, axis=1) == 0):
        return genparts[is_top]

    # children of each top: shape [events, nTop, nGenPart]
    child_mask = genparts.genPartIdxMother[:, None, :] == top_idx[:, :, None]

    # W child per top
    is_w = child_mask & (abs(genparts.pdgId)[:, None, :] == 24)
    w_idx = ak.max(ak.where(is_w, idx[:, None, :], -1), axis=2)

    # W decays through intermediate states — follow the chain to the last-copy W
    w_last = _has_flag(genparts.statusFlags, 13) & (abs(genparts.pdgId) == 24)
    w_chain_mask = genparts.genPartIdxMother[:, None, :] == w_idx[:, :, None]
    w_last_mask = w_chain_mask & w_last[:, None, :]
    w_last_idx = ak.max(ak.where(w_last_mask, idx[:, None, :], -1), axis=2)

    # quarks among the last-copy W's daughters
    w_daughter_mask = genparts.genPartIdxMother[:, None, :] == w_last_idx[:, :, None]
    is_quark = (
        w_daughter_mask
        & (abs(genparts.pdgId)[:, None, :] >= 1)
        & (abs(genparts.pdgId)[:, None, :] <= 5)
    )
    n_quarks = ak.sum(is_quark, axis=2)

    hadronic = n_quarks >= 2  # [events, nTop] bool
    n_had = ak.num(hadronic[hadronic], axis=1)

    if verbose:
        print("all-had:",  int(ak.sum(n_had == 2)))
        print("semi-lep:", int(ak.sum(n_had == 1)))
        print("di-lep:",   int(ak.sum(n_had == 0)))

    tops = genparts[is_top]
    return tops[hadronic]


def _ensure_p4(collection):
    if collection is None or "p4" in collection.fields:
        return collection
    collection["p4"] = ak.with_name(
        collection[["pt", "eta", "phi", "mass"]], "PtEtaPhiMLorentzVector"
    )
    return collection


def get_groomed_jet(jet, subjets, verbose=False):
    """Match subjets to a FatJet via dR < 0.8 and return their four-vector sum.

    Subjet association via index is unreliable in some NanoAOD versions, so we
    use a spatial match instead.
    """
    if subjets is None:
        return jet, None
    combs = ak.cartesian((jet, subjets), axis=1)
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    sel = dr_jet_subjets < 0.8
    total = combs[sel]['1'].sum(axis=1)
    return total, sel


def build_gen_top_match_info(genparts, jet0, jet1, dr_match=0.8):
    """Match two leading reco FatJets to generator-level hadronic tops.

    Only events with exactly two hadronic tops are retained; the boolean
    event_mask identifies which input events were kept.

    Returns a dict with dR values, matched-top four-vectors, and per-event
    boolean match flags (jet0_is_matched, jet1_is_matched, both_jets_matched).
    """
    tops = _ensure_p4(get_hadronic_tops(genparts))
    jet0 = _ensure_p4(jet0)
    jet1 = _ensure_p4(jet1)

    has_two_had_tops = ak.num(tops, axis=1) == 2
    tops = tops[has_two_had_tops]
    jet0 = jet0[has_two_had_tops]
    jet1 = jet1[has_two_had_tops]

    if len(tops) == 0:
        empty = ak.Array([])
        return {
            "event_mask":       has_two_had_tops,
            "tops":             tops,
            "gen_top0":         tops,
            "gen_top1":         tops,
            "jet0_dr":          empty,
            "jet1_dr":          empty,
            "jet0_is_matched":  empty,
            "jet1_is_matched":  empty,
            "both_jets_matched": empty,
            "top_pair_mass":    empty,
        }

    top_order = ak.argsort(tops.pt, ascending=False)
    tops = tops[top_order]
    gen_top0 = tops[:, 0]
    gen_top1 = tops[:, 1]

    jet0_top_pairs = ak.cartesian({"jet": ak.singletons(jet0), "top": tops}, axis=1, nested=True)
    jet1_top_pairs = ak.cartesian({"jet": ak.singletons(jet1), "top": tops}, axis=1, nested=True)
    jet0_dr_all = jet0_top_pairs["jet"].p4.delta_r(jet0_top_pairs["top"].p4)
    jet1_dr_all = jet1_top_pairs["jet"].p4.delta_r(jet1_top_pairs["top"].p4)

    jet0_match_idx = ak.flatten(ak.argmin(jet0_dr_all, axis=2), axis=1)
    jet1_match_idx = ak.flatten(ak.argmin(jet1_dr_all, axis=2), axis=1)
    jet0_dr = ak.flatten(ak.min(jet0_dr_all, axis=2), axis=1)
    jet1_dr = ak.flatten(ak.min(jet1_dr_all, axis=2), axis=1)
    jet0_gen = tops[ak.singletons(jet0_match_idx)][:, 0]
    jet1_gen = tops[ak.singletons(jet1_match_idx)][:, 0]

    jet0_is_matched = jet0_dr < dr_match
    jet1_is_matched = jet1_dr < dr_match

    return {
        "event_mask":        has_two_had_tops,
        "tops":              tops,
        "gen_top0":          gen_top0,
        "gen_top1":          gen_top1,
        "jet0_gen":          jet0_gen,
        "jet1_gen":          jet1_gen,
        "jet0_dr":           jet0_dr,
        "jet1_dr":           jet1_dr,
        "jet0_is_matched":   jet0_is_matched,
        "jet1_is_matched":   jet1_is_matched,
        "both_jets_matched": jet0_is_matched & jet1_is_matched,
        "top_pair_mass":     (gen_top0.p4 + gen_top1.p4).mass,
    }


def build_top_aligned_genjetak8_match_info(
    genparts,
    genjetak8,
    subgenjetak8,
    jet0,
    jet1,
    dr_match=0.8,
    top_align_dr=0.8,
):
    """Match reco FatJets to gen AK8 jets that are aligned with gen hadronic tops.

    Steps:
      1. Find hadronic tops; restrict to all-hadronic events.
      2. For each top, find the nearest gen AK8 jet within top_align_dr.
      3. For each reco jet, find the nearest top and inherit its aligned gen AK8 jet.
      4. A reco jet is "matched" if it is within dr_match of its top AND that
         top has a gen AK8 jet within top_align_dr.

    Returns None if genjetak8 is None or no all-hadronic events survive.
    """
    if genjetak8 is None:
        return None

    tops = _ensure_p4(get_hadronic_tops(genparts))
    genjetak8 = _ensure_p4(genjetak8)
    subgenjetak8 = _ensure_p4(subgenjetak8)
    jet0 = _ensure_p4(jet0)
    jet1 = _ensure_p4(jet1)

    has_two_had_tops = ak.num(tops, axis=1) == 2
    tops = tops[has_two_had_tops]
    genjetak8 = genjetak8[has_two_had_tops]
    if subgenjetak8 is not None:
        subgenjetak8 = subgenjetak8[has_two_had_tops]
    jet0 = jet0[has_two_had_tops]
    jet1 = jet1[has_two_had_tops]

    if len(tops) == 0:
        return None

    top_order = ak.argsort(tops.pt, ascending=False)
    tops = tops[top_order]

    top_genjet_pairs = ak.cartesian({"top": tops, "genjet": genjetak8}, axis=1, nested=True)
    dr_top_genjet = top_genjet_pairs["top"].p4.delta_r(top_genjet_pairs["genjet"].p4)

    top_genjet_match_idx = ak.argmin(dr_top_genjet, axis=2)
    top_genjet_dr = ak.fill_none(ak.min(dr_top_genjet, axis=2), 999.0)
    top_has_genjet = top_genjet_dr < top_align_dr

    aligned_genjet = genjetak8[top_genjet_match_idx]
    aligned_genjet0, _ = get_groomed_jet(aligned_genjet[:, 0], subgenjetak8)
    aligned_genjet1, _ = get_groomed_jet(aligned_genjet[:, 1], subgenjetak8)

    jet0_top_pairs = ak.cartesian({"jet": ak.singletons(jet0), "top": tops}, axis=1, nested=True)
    jet1_top_pairs = ak.cartesian({"jet": ak.singletons(jet1), "top": tops}, axis=1, nested=True)
    jet0_dr_top = jet0_top_pairs["jet"].p4.delta_r(jet0_top_pairs["top"].p4)
    jet1_dr_top = jet1_top_pairs["jet"].p4.delta_r(jet1_top_pairs["top"].p4)

    jet0_top_idx = ak.flatten(ak.argmin(jet0_dr_top, axis=2), axis=1)
    jet1_top_idx = ak.flatten(ak.argmin(jet1_dr_top, axis=2), axis=1)
    jet0_top_dr = ak.flatten(ak.min(jet0_dr_top, axis=2), axis=1)
    jet1_top_dr = ak.flatten(ak.min(jet1_dr_top, axis=2), axis=1)

    jet0_genjet = ak.where(jet0_top_idx == 0, aligned_genjet0, aligned_genjet1)
    jet1_genjet = ak.where(jet1_top_idx == 0, aligned_genjet0, aligned_genjet1)
    jet0_top_has_genjet = top_has_genjet[ak.singletons(jet0_top_idx)][:, 0]
    jet1_top_has_genjet = top_has_genjet[ak.singletons(jet1_top_idx)][:, 0]

    jet0_is_matched = (jet0_top_dr < dr_match) & jet0_top_has_genjet
    jet1_is_matched = (jet1_top_dr < dr_match) & jet1_top_has_genjet

    return {
        "event_mask":        has_two_had_tops,
        "jet0_genjet":       jet0_genjet,
        "jet1_genjet":       jet1_genjet,
        "jet0_dr_top":       jet0_top_dr,
        "jet1_dr_top":       jet1_top_dr,
        "jet0_is_matched":   jet0_is_matched,
        "jet1_is_matched":   jet1_is_matched,
        "both_jets_matched": jet0_is_matched & jet1_is_matched,
    }


# Gen merge categories of an AK8 jet, as in the top-tag T&P (toptag-sf-derivation
# production/toptag_sf_processor.py, CMS DP-2025/010): quark inside if dR(quark, jet) < 0.8.
MERGE_NOT = 0    # <= 1 quark, or only the b, or only one W quark
MERGE_SEMI = 1   # b + one W quark
MERGE_W = 2      # both W quarks, b outside
MERGE_FULL = 3   # b + both W quarks


def top_merge_category(genparts, jet_eta, jet_phi, dr_merge=0.8):
    """Per-event gen merge category (int8) of one reco jet per event.

    Same definition as the T&P that measured the top-tag SFs: b quarks whose direct
    mother is a top and light quarks whose direct mother is a W (first copies), counted
    within dr_merge of the jet axis. Events without tops (QCD) are MERGE_NOT.
    """
    pdg = abs(genparts.pdgId)
    mother = genparts.genPartIdxMother
    has_mother = mother >= 0
    mom_pdg = abs(pdg[ak.where(has_mother, mother, 0)])
    b_quarks = genparts[(pdg == 5) & (mom_pdg == 6) & has_mother]
    w_quarks = genparts[(pdg >= 1) & (pdg <= 5) & (mom_pdg == 24) & has_mother]

    def n_inside(quarks):
        dphi = (quarks.phi - jet_phi + np.pi) % (2 * np.pi) - np.pi
        dr2 = (quarks.eta - jet_eta) ** 2 + dphi ** 2
        return ak.to_numpy(ak.sum(dr2 < dr_merge ** 2, axis=1))

    b_in = n_inside(b_quarks) >= 1
    nwq = n_inside(w_quarks)
    cat = np.full(len(b_in), MERGE_NOT, dtype=np.int8)
    cat[(nwq >= 1) & b_in] = MERGE_SEMI
    cat[(nwq >= 2) & ~b_in] = MERGE_W
    cat[(nwq >= 2) & b_in] = MERGE_FULL
    return cat


# flavour code of a W-decay lepton inside a jet (w_lepton_in_jet)
WLEP_NONE, WLEP_E, WLEP_MU, WLEP_TAUH, WLEP_TAUE, WLEP_TAUMU = 0, 11, 13, 15, 1511, 1513


def w_lepton_in_jet(genparts, jet_eta, jet_phi, dr=0.8):
    """Per-event flavour code (int16) of a W-decay charged lepton inside one reco jet.

    Flags leptonic tops whose lepton lies within dr of the jet axis -- the jets that
    GloParTv3's TopbWev / TopbWmv / TopbWtauhv classes describe. Leptons whose direct
    mother is a W (first copies). A tau inside the jet is labelled by its own decay: an
    e/mu whose tau ancestor (followed through the tau copy chain) is that tau, wherever
    the e/mu lands. WLEP_NONE for hadronic tops, QCD, or a lepton outside the jet.
    No common top ancestor or b inside the jet is required (a geometric label).
    """
    pdg = abs(genparts.pdgId)
    mother = genparts.genPartIdxMother
    has_mother = mother >= 0
    safe_mother = ak.where(has_mother, mother, 0)
    mom_pdg = abs(pdg[safe_mother])

    def inside(parts):
        dphi = (parts.phi - jet_phi + np.pi) % (2 * np.pi) - np.pi
        return (parts.eta - jet_eta) ** 2 + dphi ** 2 < dr ** 2

    def any_inside(mask):
        return ak.to_numpy(ak.any(inside(genparts[mask]), axis=1))

    # first copy of every particle: walk up while the mother has the same pdgId
    first = ak.local_index(pdg)
    for _ in range(10):
        m = mother[first]
        up = (m >= 0) & (pdg[ak.where(m >= 0, m, 0)] == pdg[first])
        first = ak.where(up, m, first)

    from_w = has_mother & (mom_pdg == 24)
    is_e = any_inside(from_w & (pdg == 11))
    is_mu = any_inside(from_w & (pdg == 13))
    w_tau = from_w & (pdg == 15)
    tau_in = w_tau & inside(genparts)
    is_tau = ak.to_numpy(ak.any(tau_in, axis=1))
    tau_idx = ak.local_index(pdg)[tau_in]
    code = np.full(len(is_e), WLEP_NONE, dtype=np.int16)
    code[is_tau] = WLEP_TAUH
    for lep_pdg, lep_code in ((11, WLEP_TAUE), (13, WLEP_TAUMU)):
        # first copy of the tau each tau-decay e/mu came from
        parent = first[safe_mother][(pdg == lep_pdg) & has_mother & (mom_pdg == 15)]
        pairs = ak.cartesian([tau_idx, parent], nested=True)
        mine = ak.to_numpy(ak.any(ak.any(pairs["0"] == pairs["1"], axis=-1), axis=-1))
        code[is_tau & mine] = lep_code
    code[is_mu] = WLEP_MU
    code[is_e] = WLEP_E
    return code


# Working-point thresholds used in the diagnostic verbose block of truthstudy_counts.
# These are for exploration only — not used in any analysis selection.
_BTAG_WP = 0.5   # DeepFlavB medium WP
_TTAG_WP = 0.4   # globalParT3 loose WP


def truthstudy_counts(genparts, fatjets, subJets, jets, dr_ak8=0.8, dr_ak4=1.2, verbose=False):
    """Count gen-level hadronic tops and their geometric overlap with reco jets.

    Restricts to all-hadronic events (exactly 2 hadronic tops). Returns:
      n_hadtop          — total top slots across all all-had events
      n_hadtop_ak8      — tops matched to a FatJet within dr_ak8
      n_hadtop_ak8_ak4  — above, also having an AK4 outside the AK8 cone but within dr_ak4 of the top

    Parameters
    ----------
    verbose : bool
        Print extended diagnostic counts (top-tag rate, subjet distributions).
        Uses _BTAG_WP and _TTAG_WP thresholds — exploratory only.
    """
    tops = get_hadronic_tops(genparts, verbose=verbose)
    mask = ak.num(tops, axis=1) == 2

    tops    = tops[mask]
    fatjets = fatjets[mask]
    jets    = jets[mask]
    subJets = subJets[mask]

    if ak.sum(mask) == 0:
        return {"n_hadtop": 0, "n_hadtop_ak8": 0, "n_hadtop_ak8_ak4": 0}

    tops    = _ensure_p4(tops)
    fatjets = _ensure_p4(fatjets)
    jets    = _ensure_p4(jets)

    # match tops to AK8: [evt, nTop, nFat]
    top_fat_pairs = ak.cartesian({"top": tops, "fat": fatjets}, axis=1, nested=True)
    dr_top_fat = top_fat_pairs["top"].p4.delta_r(top_fat_pairs["fat"].p4)
    min_dr_fat = ak.fill_none(ak.min(dr_top_fat, axis=2), 999)
    has_ak8 = min_dr_fat < dr_ak8

    # AK4 jets near each top: [evt, nTop, nJet]
    top_ak4_pairs = ak.cartesian({"top": tops, "jet": jets}, axis=1, nested=True)
    dr_top_ak4 = top_ak4_pairs["top"].p4.delta_r(top_ak4_pairs["jet"].p4)
    ak4_near_top = dr_top_ak4 < dr_ak4

    # dR between all AK8 and AK4: [evt, nFat, nJet]
    fat_ak4_pairs = ak.cartesian({"fat": fatjets, "jet": jets}, axis=1, nested=True)
    dr_fat_ak4_all = fat_ak4_pairs["fat"].p4.delta_r(fat_ak4_pairs["jet"].p4)

    # pick the AK8 closest to each top, then check AK4 distance to that AK8
    closest_fat_idx = ak.argmin(dr_top_fat, axis=2)  # [evt, nTop]
    dr_fat_ak4 = ak.fill_none(dr_fat_ak4_all[closest_fat_idx], 999)

    ak4_outside_ak8 = dr_fat_ak4 > dr_ak8
    has_ak4_outside = ak.any(ak4_near_top & ak4_outside_ak8, axis=2)

    is_btag = jets.btagDeepFlavB > _BTAG_WP
    has_btag_outside = ak.any(
        ak4_near_top & ak4_outside_ak8 & is_btag[:, None, :], axis=2
    )

    if verbose:
        print("number of TRUE AK8  jets:   ", int(ak.sum(ak.ones_like(has_ak8))))
        print("number of MATCHED AK8 jets: ", int(ak.sum(has_ak8)))
        print("number of AK4 close to AK8: ", int(ak.sum(has_ak8 & has_ak4_outside)))
        print("number of AK8 + AK4(btag):  ", int(ak.sum(has_ak8 & has_btag_outside)))

        evt_has_btag = ak.any(has_btag_outside, axis=1)
        matched_fat = fatjets[closest_fat_idx]
        is_tagged = (matched_fat.globalParT3_TopbWqq > _TTAG_WP) & has_ak8
        print("matched+tagged WITH extra AK4 btag:", int(ak.sum(is_tagged & has_ak4_outside)))

        fat_sel = fatjets[evt_has_btag]
        fat_sel = fat_sel[ak.num(fat_sel, axis=1) >= 2]
        fat2 = fat_sel[ak.argsort(fat_sel.pt, ascending=False)][:, :2]
        nsub_ak8 = (
            ak.values_astype(fat2.subJetIdx1 >= 0, np.int32)
            + ak.values_astype(fat2.subJetIdx2 >= 0, np.int32)
        )
        print("nsubjets in AK8(lead, sublead) per event:", nsub_ak8[:10])
        print("mean nsubjets per AK8:", ak.mean(ak.flatten(nsub_ak8)))

    return {
        "n_hadtop":         int(ak.sum(ak.ones_like(has_ak8))),
        "n_hadtop_ak8":     int(ak.sum(has_ak8)),
        "n_hadtop_ak8_ak4": int(ak.sum(has_ak8 & has_ak4_outside)),
    }
