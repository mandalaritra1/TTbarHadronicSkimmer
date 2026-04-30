---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: coffea-dask (3.14.4)
    language: python
    name: python3
---

# ttbaranalysis

Use the configuration widgets below to choose the datasets, year, taggers, thresholds, systematics, and execution mode. The widget state is saved to `.last_config.json`, then converted into the `args` object used by the same shared analysis flow as `ttbaranalysis.py`.

```python
%load_ext autoreload
%autoreload 2

import warnings

warnings.filterwarnings("ignore")
```

```python
from python.interactive_config import build_ui, build_args
from python.run_analysis import run_analysis
```

```python
ui = build_ui()
```

```python
args = build_args(ui)
print("------args------")
for argname, value in vars(args).items():
    print(argname, "=", value)
print("----------------")
```

```python
run_summary = run_analysis(args)
```

## Standalone QCD Coffea Combiner

This cell is intentionally independent from the analysis runner above. Edit the glob pattern and output name, then run only this cell to merge the produced QCD bin outputs into one `.coffea` file.

```python
from pathlib import Path

from python.combine_coffea import combine_coffea_outputs

qcd_input_files = sorted(Path("outputs/dy").glob("QCD_2024_*_noSyst.coffea"))
qcd_output_file = "outputs/dy/QCD_2024_combined_noSyst.coffea"

if not qcd_input_files:
    raise FileNotFoundError("No QCD coffea files matched the selected pattern.")

qcd_combined = combine_coffea_outputs(qcd_input_files, qcd_output_file)

print(f"combined {len(qcd_input_files)} QCD files")
print(f"saved {qcd_output_file}")
print(qcd_combined["cutflow"])
```
