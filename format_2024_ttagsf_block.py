#!/usr/bin/env python
"""Print the ready-to-paste ``"2024": {...}`` top-tag-SF block for weights.py.

Reads the cut-and-count JSON written by ``toptag_sf_cutcount.py --out`` and
formats the tight/medium/loose WP rows into the exact literal that
``WeightsHelper._add_ttag_pt_weights`` expects (4-element nominal/up/down lists,
index 0 = <400 GeV filler, indices 1-3 = the measured [400,480],[480,600],
[600,inf] bins). Paste the printed block over the existing 2024 entry.

    python format_2024_ttagsf_block.py outputs/sf_p0.json
"""
import sys
import json


def fmt_list(xs):
    return "[" + ", ".join(f"{x:.4f}" for x in xs) + "]"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    payload = json.load(open(sys.argv[1]))
    iov = payload.get('iov', '2024')
    table = payload['sf_table'][iov]
    lumi_note = payload.get('effective_lumi_fb')
    print(f"            ## {iov} GloParTv3 top-tag SF -- MEASURED "
          f"(semileptonic ttbar tag-and-probe, mu channel, merge-corrected")
    print(f"            ## cut-and-count; CMS DP-2025/010 method). "
          f"up/down = nominal +/- stat error." + (
              f" Effective lumi {lumi_note} fb^-1 (P0 subset)." if lumi_note else ""))
    print(f'            "{iov}": {{')
    for wp in ('tight', 'medium', 'loose'):
        row = table[wp]
        print(f'                "{wp}": {{"nominal": {fmt_list(row["nominal"])}, '
              f'"up": {fmt_list(row["up"])}, "down": {fmt_list(row["down"])}}},')
    print('            },')


if __name__ == '__main__':
    main()
