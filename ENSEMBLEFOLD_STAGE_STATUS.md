# EnsembleFold wrapper stage status — 2026-09-23

Defaults: `write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Data prepares user-provided inputs; it does not
search MSA databases or support structure templates. Native keyed MSA and split
paired/unpaired MSA remain separate supported modes, never mixed within one
complex. See `ESMFOLD2_WRAPPER.md` for inputs and result layouts.

CPU/offline wrapper regression: 158 tests; no GPU/model equivalence claim.
There was no template-declaration change in this repository.

Known deferred limitation: an interrupted replacement can leave new structures
with old confidence files which still pass existence/non-empty skip checks.
Compression controls do not make publication a whole-sample transaction. Skip
does not compare scientific input conditions; use a fresh output directory or
disable skip to require a new prediction.
