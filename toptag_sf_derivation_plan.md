# Top-tag efficiency SF derivation — moved to its own project

The scale-factor measurement (semileptonic ttbar tag-and-probe for the
GloParTv3 `TopvsQCD` WPs) is now a standalone project:

**`~/Projects/toptag-sf-derivation/`** — canonical `PLAN.md` there.

It stays coupled to this repo: the `TopTagSFProcessor` code is expected to land
in `python/` here, it reuses `python/truthstudy.py` (`get_hadronic_tops`) and the
`_tscore` formula (`ttbarprocessor.py:272`), and its output fills
`python/weights.py` `ttag_scale_factors[iov][wp]` (placeholders today). See the
companion WP-threshold plan `toptag_wp_derivation_plan.md` in this repo.
