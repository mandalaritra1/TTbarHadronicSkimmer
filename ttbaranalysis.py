# ttbaranalysis.py

import argparse
import warnings

warnings.filterwarnings("ignore")

DEFAULT_DATASETS = ["data", "TTbar", "QCD"]
DEFAULT_SIGNALS = ["RSGluon", "ZPrime10", "ZPrime30", "ZPrimeDM", "ZPrime1"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ttbaranalysis.py",
        description="Run ttbarprocessor",
        epilog="e.g. python ttbaranalysis.py --test --dataset TTbar --iov 2024",
    )

    parser.add_argument(
        "-d", "--dataset",
        choices=[
            "data", "QCD", "QCD_flat", "TTbar", "ZPrime1", "ZPrime10",
            "ZPrime30", "ZPrimeDM", "RSGluon", "ZPrimeLocal",
        ],
        default=DEFAULT_DATASETS,
        action="append",
    )
    parser.add_argument("--iov", choices=["2022", "2023", "2024"], default="2024")
    parser.add_argument("--signals", action="store_true", help="run only signal samples")

    parser.add_argument("--era", choices=list("ABCDEFGHI"), action="append", default=[])
    parser.add_argument("-p", "--pt", choices=["700to1000", "1000toInf"], action="append", default=[])
    parser.add_argument("-m", "--mass", action="append", default=[])
    parser.add_argument("--subsample", action="append", default=[])

    parser.add_argument("--blind", action="store_true", help="process 1/10th of the data")
    parser.add_argument("--bkgest", choices=["2dalphabet", "mistag"], default=None)
    parser.add_argument("--toptagger", choices=["deepak8", "cmsv2"], default="deepak8")
    parser.add_argument("-r", "--redirector", default="root://cmsxrootd.fnal.gov/")
    parser.add_argument("--ttagWP", choices=["loose", "medium", "tight"], default="medium")
    parser.add_argument("--btagger", choices=["deepcsv", "csvv2"], default="deepcsv")
    parser.add_argument("--ht", choices=["1400", "950"], default="1400")
    parser.add_argument("--noSyst", action="store_true", help="run without systematics")
    parser.add_argument("--ntuple", action="store_true", help="collect flat ntuple in output")
    parser.add_argument("--ntupleContent", choices=["slim", "full"], default="slim")
    parser.add_argument("--ntupleStorage", choices=["chunks", "accumulator"], default="chunks")
    parser.add_argument("--ntupleBaseDir", default="")

    parser.add_argument("--dask", action="store_true")
    parser.add_argument("--env", choices=["casa", "lpc", "winterfell", "local", "C", "L", "W"], default="lpc")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-n", "--nocluster", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--daskMemory", type=int, default=5)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="events per coffea chunk; 0 uses the default for the selected mode",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if len(args.dataset) > len(DEFAULT_DATASETS):
        args.dataset = args.dataset[len(DEFAULT_DATASETS):]
    if args.signals:
        args.dataset = DEFAULT_SIGNALS

    from python.run_analysis import run_analysis

    run_analysis(args)
