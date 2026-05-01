# PKL/Coffea Explorer

## Static Command Builder

`apps/command_builder.html` is a static browser-only helper for building the
equivalent `ttbaranalysis.py` command from form selections. It can be opened
directly from disk or through a static file CDN. Its dataset/subsample choices
are hard-coded from the current `data/nanoAOD/*.json` manifests.

GitHub's normal file view displays HTML source instead of running it. Use a raw
static-file URL, for example:

```text
https://cdn.jsdelivr.net/gh/mandalaritra1/TTbarHadronicSkimmer@reorg/cleanup/apps/command_builder.html
```

## Coffea Explorer

Launch from the repository root:

```bash
streamlit run apps/coffea_explorer.py
```

The app discovers `.coffea`, `.pkl`, and `.pickle` files under `outputs/` and also accepts manual paths. It can inspect top-level keys, plot `hist.Hist` objects, show cutflows and analysis categories, and plot embedded `output["ntuple"]` branches when present.

Use **File slots** to load up to three files. In the histogram tab, use **Traces** to overlay compatible 1D projections, such as `data` and `ttbar` from the same pickle file, or to compare matching histograms from different files. The **Axis ranges** expander slices numeric axes before projection, so you can plot `ttbarmass` after restricting `jetmass` to a signal window.
