# Top-tag efficiency SF derivation — lives in its own project

The scale-factor measurement (semileptonic ttbar tag-and-probe for the
GloParTv3 `TopvsQCD` WPs) is the standalone project
**`~/Projects/toptag-sf-derivation/`** (canonical `PLAN.md` there). Its code
(processor, runners, fits) is only there; the stale copy that used to sit in this
repo was removed in v1.2 (recover it from `7b51c08` if ever needed).

Its results reach this repo as a versioned table,
`data/toptag/ttag_sf_<version>.json`, read by `python/weights.py`
(`TTAG_SF_FILE`). Each 2024 entry records the T&P commit and result file it came
from. See the companion WP-threshold plan `toptag_wp_derivation_plan.md`.
