# EnsembleFold-compatible ESMFold2 wrapper

This first-layer wrapper gives ESMFold2 inspectable JSON/MSA input and the
EnsembleFold seed/sample output layout. Its lightweight data pipeline validates
and organizes existing inputs; optional snapshot publication is separate from inference.

The wrapper does **not** run an MSA search. A protein with no MSA fields is
folded in query-only mode. To use MSA conditioning, provide either a native A3M
or Boltz-style paired and unpaired A3Ms. The full `biohub/ESMFold2` checkpoint
supports MSA conditioning; a Fast checkpoint rejects inputs containing an MSA.
The older Experimental architecture is not supported because it does not
produce the PDE output required by the wrapper's fixed output contract.

## Installation and entry points

Install the checkout into the active Python 3.12+ environment:

```bash
pip install -e .
```

The installed command and source-checkout module are equivalent:

```bash
esmfold2-wrapper predict --help
python -m esm.esmfold2_wrapper predict --help
```

The convenience shell wrapper uses the active `python`, or the executable set
in `PYTHON_BIN`:

```bash
./run_esmfold2.sh -h
```

Local inference is designed for one visible CUDA GPU. CPU is accepted by the
Python CLI but is generally impractical for the full checkpoint. ROCm uses the
PyTorch `cuda` device spelling.

## Input JSON

Every input is one versioned JSON object:

```json
{
  "version": 1,
  "name": "target",
  "sequences": [
    {
      "type": "protein",
      "id": "A",
      "sequence": "MKTAYIAKQRQISFVKSHFSRQ"
    }
  ]
}
```

`name` and every chain ID must use ASCII letters, digits, `_`, `-`, or `.`.
Polymer sequences use uppercase ASCII residue letters. An `id` can be a string
or a list such as `["A", "B"]` to instantiate identical homomer copies. To give
identical sequences different MSAs, use separate protein entries.

Supported entities are:

- `protein`: `id`, `sequence`, optional `modifications`, and one optional MSA mode.
- `rna` or `dna`: `id`, `sequence`, and optional `modifications`.
- `ligand`: `id` and exactly one of `ccd` or `smiles`.

Polymer modifications are zero-indexed:

```json
{"position": 10, "ccd": "MSE"}
```

Chain-break characters `:` and `|` are not accepted. Represent each chain as a
separate entity.

### Native MSA mode

Set exactly one of inline `msa` or filesystem `msaPath`:

```json
{
  "type": "protein",
  "id": "A",
  "sequence": "ACDE",
  "msaPath": "inputs/A.a3m.zst"
}
```

MSA files may be plain UTF-8, gzip, xz, or zstd; compression is detected from
magic bytes rather than the filename suffix. Relative paths are resolved from
the JSON file.

Native ESMFold2 pairing keys can appear in A3M headers as `key=N`. Valid values
are `-1` for an unpaired row or a non-negative paired-row key.
Writing an input snapshot preserves these headers and keys; it never splits a
native MSA into paired and unpaired files.

### Paired plus unpaired MSA mode

Set both a paired and an unpaired source. Each source independently uses an
inline field or a path field:

```json
{
  "type": "protein",
  "id": "A",
  "sequence": "ACDE",
  "pairedMsaPath": "boltz/A_paired.a3m.zst",
  "unpairedMsaPath": "boltz/A_unpaired.a3m.zst"
}
```

The accepted format matches the paired and unpaired component A3Ms produced by
the Boltz wrapper. Conversion to ESMFold2's keyed MSA follows these rules:

- A paired row receives its original zero-based row position as `key=N`.
- Boltz all-gap paired padding rows are removed without renumbering later rows.
- Unpaired rows receive `key=-1`.
- When paired rows exist, the duplicate query at the start of the unpaired MSA
  is omitted.
- An explicitly empty paired MSA is allowed; its unpaired query remains first.
- Split paired files across protein entities must have the same original row
  count so equal keys describe the same paired sequence row.

Lowercase A3M insertions are removed from aligned sequences while their deletion
features are retained by the native ESMFold2 MSA parser.

Within one complex, all proteins carrying MSA input must use the same mode:
native or paired/unpaired. Mixing modes across chains raises an error listing
the chain IDs. Query-only proteins and non-protein entities do not choose a mode.
Split snapshots retain the supplied A3M records, including all-gap padding rows;
the row-number keys are generated only in memory for inference.

## Data pipeline, inference, and optional input snapshot

`-D/--run-data-pipeline` and `-P/--run-inference` select the stages; both default
to true. `-J/--write-input-json` (also `--write_input_json` in the Python CLI)
independently controls snapshot publication and defaults to **true**.
At least one stage must be enabled. Inference-only and fully skipped runs refresh
current input resources when publication is enabled; no search is introduced.

`--compress-fold-input` / `--compress_fold_input` (shell `-z`) defaults to false:
MSAs remain external plain text. True writes zstd resources. Readers accept either
format regardless of this write setting. `--compress-full-confidence` /
`--compress_full_confidence` (shell `-f`) defaults to false: separate pLDDT, PAE
and PDE files use JSON. True uses compressed NPZ with the same stems, keys, values
and array shapes. Switching format removes old counterparts after writing. Seed
embeddings remain NPZ, and skip checks use the selected confidence format.

The data stage reads and validates supplied JSON/MSAs, including query matches
and cross-chain paired depths. It neither searches nor builds model features.
With J=false it validates without writing public files. To save input without
loading a model or running inference:

```bash
./run_esmfold2.sh -i target.json -o results -D true -P false -J true
```

Predict and save a portable copy of the supplied conditions:

```bash
./run_esmfold2.sh \
  -i target.json \
  -o results \
  -J true \
  -r 101
```

In addition to predictions, this validates and writes a portable input snapshot:

```text
results/target/
├── target_data.json
└── msas/
    ├── target__A_pairedmsa.a3m
    └── target__A_unpairedmsa.a3m
```

Native input instead produces `msas/target__A_msa.a3m`. The JSON uses paths
relative to itself, even when the supplied paths were absolute or external.
Files are updated even if a previous snapshot exists or every seed is skipped.
No MSA search, native-to-split conversion, or feature cache is involved.
Arbitrary old input filenames remain readable; these names only govern new output.
Snapshot files are individually atomically replaced, with JSON published last;
this is not a cross-file transaction. If a write is interrupted, refresh the
snapshot from the original input before using it. Do not run concurrent writers
against the same job snapshot.

Predict from that snapshot without rewriting it:

```bash
./run_esmfold2.sh \
  -i results/target/target_data.json \
  -o results \
  -D false -P true \
  -J false \
  -r 101,102,103 \
  -n 5 \
  -S true
```

For fully local weights, pass the ESMFold2 and ESMC directories separately.
Set `ESMCFOLD_CCD_PATH` to the CCD file distributed with the ESMFold2
checkpoint so input preparation is offline as well:

```bash
export ESMCFOLD_CCD_PATH=/path/to/ESMFold2/ccd.pkl
./run_esmfold2.sh \
  -i results/target/target_data.json \
  -o results \
  -J false \
  -k /path/to/ESMFold2 \
  -m /path/to/ESMC-6B
```

The installed CLI has the same behavior:

```bash
esmfold2-wrapper predict \
  -i target.json \
  -o results \
  --write-input-json true \
  -r 101 \
  -n 5
```

If no seed is supplied for inference, the wrapper generates and reports one
concrete uint32 seed. Multiple seeds retain their requested order. The model and
input builder are loaded once per invocation and reused across pending seeds.
Every invocation with pending seeds reads its selected JSON and current MSA
resources afresh, then builds native features during inference. `-J false` does
not create or rewrite public JSON/MSA resources, even if old ones exist under the
output root. It still writes prediction results.
Explicit J=true also works during inference-only (D=false/P=true): it validates
and materializes the supplied resources but never starts a search. Data-only
returns no model results and ignores inference-specific seed/sampling options.

## Core inference options

| Option | Default | Meaning |
| --- | ---: | --- |
| `-D, --run-data-pipeline` | `true` | Read/validate/organize existing input, without searching |
| `-P, --run-inference` | `true` | Run prediction for pending seeds |
| `-J, --write-input-json` | `false` | Write/update portable input JSON and MSA resources |
| `-r, --seeds` | generated | One uint32 seed or comma-separated seeds |
| `-n, --diffusion-samples` | `5` | Samples per seed |
| `-c, --loops` | `20` | ESMFold2 folding loops |
| `-p, --sampling-steps` | `200` | Diffusion sampling steps |
| `-k, --checkpoint` | `biohub/ESMFold2` | Local path or Hugging Face ID |
| `--esmc-checkpoint` (`-m` in shell script) | checkpoint configuration | Local ESMC directory, overriding the ESMFold2 config's Hub ID |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N` |
| `--dtype` | `auto` | Trunk weights: checkpoint precision or FP32 |
| `--esmc-precision` | `bf16` | Independent ESMC precision: FP32, BF16, or FP8 |
| `--kernel-backend` | `auto` | Reference/default, fused, or cuequivariance |
| `--lm-dropout` | `0.3` | Inference LM dropout; use `0` to disable |
| `--msa-max-depth` | `1024` | Maximum MSA depth |
| `--msa-column-mask-rate` | `0.1` | Fraction of MSA columns masked at inference |
| `-E, --include-embeddings` | `false` | Write seed-level pooled pair embeddings |
| `-S, --skip` | `false` | Skip seeds whose required canonical files exist and are nonempty |

`--dtype auto` is recommended: it preserves checkpoint precision while the CUDA
forward path uses its native autocast behavior. FP8 ESMC requires compatible
hardware and Transformer Engine. Optional fused/cuequivariance backends require
their matching dependencies.

## Outputs

Sample indices are zero-based and retain native diffusion order; samples are not
ranked or renamed:

```text
results/target/
├── models/
│   └── seed-101_sample-0_model.cif
├── summary_confidences/
│   └── seed-101_sample-0_summary_confidences.json
├── full_data/
│   ├── plddt_seed-101_sample-0.json
│   ├── pae_seed-101_sample-0.json
│   └── pde_seed-101_sample-0.json
└── embeddings/
    └── seed-101_embeddings.npz
```

The mmCIF contains all supported polymer, ligand, modification, chain, and
coordinate information. Its B-factor column contains structure-token pLDDT on
the 0–100 scale.

The pLDDT NPZ contains:

- `plddt`: model-token pLDDT on the 0–1 scale, aligned with PAE/PDE axes.
- `structure_token_plddt`: pLDDT on the 0–1 scale, aligned with serialized mmCIF
  tokens after modified-residue and ligand-token collapsing.

The summary JSON contains `seed`, `sample`, `mean_plddt`, `ptm`, `iptm`, ordered
`chain_ids`, and a chain-ID-keyed `pair_chains_iptm` matrix. The optional
embedding file is shared by all diffusion samples for that seed.

All returned samples are validated before writing starts. As in AF3 Pro, the
canonical model/full-data files are written directly and summaries are written
last; there is no cross-file transaction or rollback.

## Skip behavior

With `--skip true`, a seed is skipped only when every requested sample has its
canonical model, summary, pLDDT, PAE, and PDE file as a nonempty regular file.
When embeddings are requested, the seed-level embedding file must also be
nonempty. The check only uses file metadata; it does not read or parse contents.
If any required file is missing or empty, the whole seed is rerun with all
requested samples. Complete seeds are preserved.

Skip does not compare the prepared input, MSA, checkpoint, loop count, sampling
steps, dropout, or other inference settings. Changed conditions can still skip;
use a new output root or disable skip when new predictions are wanted. Existing
optional embeddings and extra sample indices are left untouched and ignored
when they are not requested.

## Current first-layer scope

The wrapper supports proteins, RNA, DNA, CCD/SMILES ligands, polymer
modifications, query-only folding, native MSA input, and Boltz-style split MSA
input. It intentionally does not expose MSA search, templates, covalent-bond
constraints, pocket conditioning, custom CCD definitions, ranking, or
distributed/context-parallel inference.
