# Repository Agent Instructions

This is an active CMS analysis code repository. You may edit this repository when the user asks for code work.

For analysis context, first read:

- `/Users/aritra/Projects/ai-wiki/AGENTS.md`
- `/Users/aritra/Projects/ai-wiki/wiki/meta/index.md`

Then open the repo-specific wiki card:

- `/Users/aritra/Projects/ai-wiki/wiki/repos/ttbarhadronic_skimmer.md`

Treat `/Users/aritra/Projects/ai-wiki` as the compiled knowledge base. When a code change creates durable analysis knowledge, changes workflow behavior, fixes a reusable bug, or changes project status, you may edit `/Users/aritra/Projects/ai-wiki` to update the relevant repo, topic, bug, project, synthesis, or question pages.

When editing the wiki, follow `/Users/aritra/Projects/ai-wiki/AGENTS.md`, update `wiki/meta/index.md` if pages are created, removed, renamed, or meaningfully changed, and append `wiki/meta/log.md`. Treat raw sources, PDFs, research notes, and other linked repositories as read-only unless the user explicitly asks otherwise.

Physics findings and results produced from work in this repository (tagging studies, background-estimate results, fit/limit outcomes, etc.) should also be logged to the research-notes vault at `/Users/aritra/Projects/research-notes`: promote findings into its `topics/`/`bugs/` and update `projects/ttbarhadronic.md`, following that vault's own `AGENTS.md`. Include plots whenever possible — save the PNG into `research-notes/attachments/` and embed it in the note with a relative `../attachments/...` link and descriptive alt text.

Log proactively. Whenever a session involves testing, code changes, debugging, or produces a result, record the durable outcome at natural checkpoints (a finding confirmed, a bug root-caused, a feature validated) — to `ai-wiki` for code/workflow/debugging knowledge and to `research-notes` for physics findings — not only when explicitly asked.

Additional project documentation lives at `/Users/aritra/Projects/documentations/doc-ttbarhad`. Treat it as documentation context and follow its own `AGENTS.md` before editing it.

## Downstream: 2DAlphabet fit stage (bgestimation)

This repo is the **upstream** skimmer. Its outputs feed the **downstream**
2DAlphabet background-estimation + Combine fit repo `mandalaritra1/bgestimation`
(local `/Users/aritra/Projects/bg_ttbar/src/bgestimation`; card
`/Users/aritra/Projects/ai-wiki/wiki/repos/ttbarhadronic_bgestimation.md`).

Handoff: `plots/make2Drootfiles.py` (+ `plots/scaleCoffeaFiles.ipynb`) converts the
coffea `mtt_vs_mt` outputs into 2DAlphabet TH2 inputs `MttvsMt{Cen,Fwd}24{Pass,Fail}`
in `TTbarAllHad24_*.root`, staged on EOS at
`/eos/user/a/amandal/ttbarhad_root_files/2dAlphabetInputs`. Those histogram names
are the **interface contract** — if the region split or binning changes here, update
the bgestimation 2024 configs too (and vice versa).
