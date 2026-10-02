"""Integrated luminosity per IOV [pb^-1]: the one table the processor, the
normalization helpers and the plot scripts read (a copy in functions.py had
kept 2025 at 110.59 fb^-1 after the processor moved to 110.37)."""

LUMI_PB = {
    '2016APV': 19800.,
    '2016':    16120.,
    '2016all': 35920.,
    '2017':    41530.,
    '2018':    59740.,
    '2023':    27000.,
    '2024':    109950.,  # golden-JSON certified 2024 lumi (109.95 fb^-1)
    # 2025 prompt-reco (NanoAODv15), eras C-G (PPD excludes era B). Reference: the
    # PPD Run3-2025 table on the PdmV Run-3 analysis TWiki (updated as the golden
    # JSON evolves; version dated 20 Jan 2026): 21.56+25.82+14.05+26.69+22.25 =
    # 110.37 fb^-1. Reproduced 2026-09-24 with brilcalc --normtag normtag_BRIL on
    # Cert_Collisions2025_391658_398903_Golden.json. Re-check whenever either changes.
    '2025':    110370.,
    # 2026 PromptReco, eras B + D (nominal PU). PdmV Run-3 TWiki 2026 table (r223,
    # 2026-09-07, preliminary golden-JSON values): B 15.28 + D 10.03 = 25.31 fb^-1.
    # Era A (0.64, first collisions, not in the PPD analysis table) and the low-PU era C
    # (2.11) are not used. Golden JSON Cert_Collisions2026_401624_403937_golden.json.
    '2026':    25310.,
    # Run-3 sub-era keys (NanoAODv15). Preliminary golden-JSON values; refine
    # with brilcalc on data/corrections/goldenJsons/.
    '2022preEE':    7980.,   # Run2022 C,D
    '2022postEE':   26670.,  # Run2022 E,F,G
    '2023preBPix':  17794.,  # Run2023 B,C
    '2023postBPix': 9451.,   # Run2023 D
}
