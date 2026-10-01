#!/usr/bin/env python3
"""Build the skimmer manifest for the private 2022/2023 NanoAODv15 Z' signal.

The 1%-width Z' samples for 2022/2023 exist centrally only as NanoAODv12 (no GloParTv3),
so M = 2, 4, 6 TeV were re-NANO'd privately from the MiniAODv4 parents on lxplus
(CMSSW_15_0_15_patch4, the central v15 era/GT per campaign; validated bit-identical to
the central v15 TTto4Q on 200 events). Output layout on CERNBox:
    <base>/ZPrimeToTT_M<mass>_W<mass/100>/<campaign>/*.root

The manifest has the same shape as data/nanoAOD/ZPrime1.json ({iov: {mass: [files]}}),
with EOS paths, so the skimmer reads it with the CERNBox redirector:
    python ttbaranalysis.py -d ZPrime1 --manifest data/nanoAOD/ZPrime1_renano.json \\
        -r root://eosuser.cern.ch/ --iov 2022preEE -m 2000 -m 4000 -m 6000 ...
"""
import argparse
import json
import re
import subprocess

CAMPAIGN_TO_IOV = {
    "Summer22": "2022preEE",
    "Summer22EE": "2022postEE",
    "Summer23": "2023preBPix",
    "Summer23BPix": "2023postBPix",
}


def local_ls(path):
    import os
    return [os.path.join(path, f) for f in sorted(os.listdir(path))]


def xrdfs_ls(server, path):
    if server is None:
        return local_ls(path)
    out = subprocess.run(["xrdfs", server, "ls", path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"xrdfs ls {path}: {out.stderr.strip()}")
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="root://eosuser.cern.ch")
    ap.add_argument("--base", default="/eos/user/a/amandal/ttbarhadronic/renano_v15")
    ap.add_argument("--local", metavar="DIR",
                    help="list a locally synced copy instead of EOS (e.g. ~/cernbox/ttbar/renano_v15); "
                         "the manifest then holds local paths: run the skimmer with -r ''")
    ap.add_argument("--jobs", help="lxplus jobs.txt (campaign sample lfn skip nevents) to check completeness")
    ap.add_argument("-o", "--out", default="data/nanoAOD/ZPrime1_renano.json")
    args = ap.parse_args()
    if args.local:
        import os
        args.server, args.base = None, os.path.expanduser(args.local)

    expected = {}
    if args.jobs:
        for line in open(args.jobs):
            campaign, sample = line.split()[:2]
            expected[(sample, campaign)] = expected.get((sample, campaign), 0) + 1

    manifest = {}
    complete = True
    for sample_dir in sorted(xrdfs_ls(args.server, args.base)):
        sample = sample_dir.rstrip("/").split("/")[-1]
        m = re.fullmatch(r"ZPrimeToTT_M(\d+)_W(\d+)", sample)
        if not m or int(m.group(2)) * 100 != int(m.group(1)):
            continue  # 1% width only
        mass = m.group(1)
        for campaign_dir in sorted(xrdfs_ls(args.server, sample_dir)):
            campaign = campaign_dir.rstrip("/").split("/")[-1]
            iov = CAMPAIGN_TO_IOV.get(campaign)
            if iov is None:
                continue
            files = sorted(f for f in xrdfs_ls(args.server, campaign_dir) if f.endswith(".root"))
            manifest.setdefault(iov, {})[mass] = files
            exp = expected.get((sample, campaign))
            flag = "" if exp is None or exp == len(files) else f"  INCOMPLETE (expected {exp})"
            complete &= not flag
            print(f"{iov:13s} M{mass:5s} {len(files):3d} files{flag}")

    manifest = {iov: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for iov, v in sorted(manifest.items())}
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}" + ("" if complete else "  -- some samples are incomplete"))


if __name__ == "__main__":
    main()
