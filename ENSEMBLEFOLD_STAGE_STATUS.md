# EnsembleFold wrapper stage status — 2026-09-24

Defaults: `write_input_json=true`, `compress_fold_input=false`, and
`compress_full_confidence=false`. Data prepares user-provided inputs; it does not
search MSA databases or support structure templates. Native keyed MSA and split
paired/unpaired MSA remain separate supported modes, never mixed within one
complex. See `ESMFOLD2_WRAPPER.md` for inputs and result layouts.

CPU/offline wrapper regression: 158 tests; no GPU/model equivalence claim.
There was no template-declaration change in this repository.

Final verification (2026-09-24): 158 CPU wrapper tests passed again (job
93948). Real single-GPU inference in allocation 91413 passed with the existing
ESMFold2 and ESMC-6B weights, 20 residues, query-only MSA, seed 101, one sample,
one loop and five diffusion steps. CIF, prepared input, summary and full-data
JSON files were checked. This is a small end-to-end smoke, not a native/default-
kernel numerical-equivalence claim; native optional-kernel fallbacks were logged.
Corrected the option table's stale write-input-json default to true; code is unchanged.

Known deferred limitation: an interrupted replacement can leave new structures
with old confidence files which still pass existence/non-empty skip checks.
Compression controls do not make publication a whole-sample transaction. Skip
does not compare scientific input conditions; use a fresh output directory or
disable skip to require a new prediction.
