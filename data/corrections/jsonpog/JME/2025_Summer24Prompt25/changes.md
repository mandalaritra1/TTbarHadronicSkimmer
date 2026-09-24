## Changes: 2026-07-16 (Update to \[Summer24Prompt25_V3\] JEC tag)

Merge Request: [!4](https://gitlab.cern.ch/cms-analysis-corrections/JME/Run3-25Prompt-Summer24-NanoAODv15/-/merge_requests/4)

This MR fixes a bug in the recent `2026-07-14` release where the `L2L3Residual` corrections for 2025 were not properly updated. Despite the tag transition (`Summer24Prompt25_V1` $\rightarrow$ `Summer24Prompt25_V2`), the underlying JEC payloads incorrectly remained identical to their older versions.

To resolve this issue, we introduce the corrected payloads under a new tag:

- 2025 Analyses: `Summer24Prompt25_V3_MC` and `Summer24Prompt25_V3_DATA` (replacing V2)

This new version now correctly contains the updated `L2L3Residual` corrections as intended.
