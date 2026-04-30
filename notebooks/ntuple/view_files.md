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

```python
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# add any other imports you need here
# Set the file to inspect — swap in any path from qcd.json
filename = "/store/mc/Run3Winter24NanoAOD/QCD_PT-15to7000_TuneCP5_Flat2022_13p6TeV_pythia8/NANOAODSIM/JMENanoV14_133X_mcRun3_2024_realistic_v10-v1/2530000/01eebb7d-8f73-4e7a-baae-83c9ff84af44.root"
filename = "/store/mc/RunIII2024Summer24NanoAODv15/QCD_Bin-PT-15to7000_Par-PT-Flat_TuneCH3_13p6TeV_herwig7/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/2820000/005f8b3c-b044-48c0-bf57-77db0939ade4.root"
redirector = "root://cms-xrd-global.cern.ch//"

events = NanoEventsFactory.from_root(
    {redirector + filename: "Events"},
    schemaclass=NanoAODSchema,
    entry_stop=1,  # only need 1 event to read schema
).events()
```

```python
print("=== Top-level event fields ===")
print(sorted(events.fields))
```


```python
print("=== FatJet fields ===")
for f in sorted(events.FatJet.fields):
    print(f"  FatJet_{f}")
```


```python
print("=== FatJet tagger-related fields ===")
keywords = ["top", "ttag", "particleNet", "globalParT", "deepTag", "ParT", "Xtt", "Top"]
for f in sorted(events.FatJet.fields):
    if any(k.lower() in f.lower() for k in keywords):
        print(f"  FatJet_{f}")
```


```python

```

## TTbar gen weight and filter fields

```python
import json

with open("data/nanoAOD/TTbar.json") as f:
    ttbar_json = json.load(f)

filename = ttbar_json["2024"]["inclusive"]["files"][0]
redirector = "root://cms-xrd-global.cern.ch/"
full_filename = redirector + filename

print(full_filename)
print(ttbar_json["2024"]["inclusive"]["metadata"])
```

```python
events = NanoEventsFactory.from_root(
    {full_filename: "Events"},
    schemaclass=NanoAODSchema,
    entry_stop=10,
).events()
```

```python
print("=== Event fields with weight/filter names ===")
for f in sorted(events.fields):
    if "weight" in f.lower() or "filter" in f.lower():
        print(f)
```

```python
print("genWeight first events:")
print(events.genWeight.to_numpy())

if "LHEWeight" in events.fields:
    print("LHEWeight fields:")
    print(events.LHEWeight.fields)

if "LHEWeight_originalXWGTUP" in events.fields:
    print("LHEWeight_originalXWGTUP first events:")
    print(events.LHEWeight_originalXWGTUP.to_numpy())
```

```python
import uproot

with uproot.open(full_filename) as f:
    print("=== Trees ===")
    print(f.keys())

    print("\n=== Runs branches with gen/sumw/filter ===")
    runs = f["Runs"]
    for branch in runs.keys():
        if "gen" in branch.lower() or "sumw" in branch.lower() or "filter" in branch.lower():
            print(branch)

    print("\n=== Runs stored values ===")
    for branch in ["genEventCount", "genEventSumw", "genEventSumw2"]:
        if branch in runs.keys():
            print(branch, runs[branch].array(library="np"))

    print("\n=== LuminosityBlocks branches with GenFilter ===")
    lumis = f["LuminosityBlocks"]
    for branch in lumis.keys():
        if "genfilter" in branch.lower():
            print(branch)

    print("\n=== LuminosityBlocks GenFilter stored values ===")
    for branch in [
        "GenFilter_filterEfficiency",
        "GenFilter_filterEfficiencyError",
        "GenFilter_numEventsPassed",
        "GenFilter_numEventsTotal",
    ]:
        if branch in lumis.keys():
            print(branch, lumis[branch].array(library="np"))
```

```python

```
