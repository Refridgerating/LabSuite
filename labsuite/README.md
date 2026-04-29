# LabSuite Operator Guide

LabSuite is a reproducible scientific analysis platform for ESR, FMR, and VSM
measurements. The supported operator surface is the `labsuite` CLI. The GUI
entry point is declared in package metadata, but it is still scaffold-level and
is not the recommended workflow for analysis.

Run examples from this `labsuite/` package directory:

```powershell
cd F:\LabAnalysisSuite\labsuite
```

## Setup

LabSuite targets Python 3.11+.

```powershell
python -m pip install -e .
labsuite --version
```

For development extras and tests:

```powershell
python -m pip install -e ".[dev]"
pytest
```

All paths below are relative to this `labsuite/` directory.

## Pipeline Contract

Every implemented workflow follows:

```text
parse -> preprocess -> fit -> derive -> export
```

Parsing reads raw files and preserves metadata. Preprocessing performs explicit,
recipe-driven cleanup. Fitting applies modality-specific scientific models.
Derived parameters are calculated after fitting. Exports write reproducible
artifacts and provenance to disk.

Scientific logic belongs in `src/labsuite/core/`,
`src/labsuite/plugins/<modality>/`, `src/labsuite/workflows/`, or
`src/labsuite/sample_analysis/`. GUI code should trigger services and render
results only.

## CLI Map

Primary commands:

```text
labsuite --version
labsuite esr single|batch|config|export|report
labsuite vsm single|batch|config|export|report
labsuite fmr single|batch|config|export|report
labsuite sample add|update|list|show|register-file|validate|readiness|analyze|analyze-batch
```

Compatibility commands:

```text
labsuite fit-single
labsuite fit-batch
labsuite esr-single
```

`fit-single`, `fit-batch`, and `esr-single` are ESR-only compatibility aliases.
Prefer `labsuite esr single` and `labsuite esr batch` for new work.

`labsuite-gui` exists as a package script, but the CLI and backend are currently
the supported analysis path.

## Raw Modality Analysis

Single-file processing:

```powershell
labsuite esr single `
  --input data/raw/esr_sample.dsc `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_single `
  --fit-mode auto

labsuite fmr single `
  --input "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_single

labsuite vsm single `
  --input data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_single
```

Batch processing:

```powershell
labsuite esr batch `
  --input data/raw/ESR `
  --pattern "*.dsc" `
  --recursive `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_batch `
  --fit-mode auto

labsuite fmr batch `
  --input data/raw/FMR `
  --pattern "*.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_batch

labsuite vsm batch `
  --input data/raw/VSM/MTJ-NoPattern `
  --pattern "*.dat" `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_batch
```

Single commands accept a direct source file or a directory that resolves to
exactly one source. Batch commands accept a file or folder. If `--output-dir` is
omitted, single runs write to `data/processed/<source-stem>/` and batch runs
write to `data/processed/<input-name>_<timestamp>/`.

Default discovery patterns are `.dsc` for ESR, `.log` for FMR, and `.dat` for
VSM.

## Recipes

Print or write a default recipe:

```powershell
labsuite esr config
labsuite fmr config --output recipes/fmr/my_run.yaml
labsuite vsm config --output recipes/vsm/my_run.yaml
```

Default recipes:

- `recipes/esr/default.yaml`
- `recipes/fmr/default.yaml`
- `recipes/vsm/default.yaml`
- `recipes/sample_analysis/default.yaml`

For tuned work, copy a default recipe and pass the copy with `--recipe`. Result
folders include recipe and metadata snapshots so the run can be reproduced.

## Registry-Aware Analysis

The metadata stores live in `metadata/`:

- `sample_registry.yaml`: physical sample records and defaults.
- `measurement_ledger.yaml`: stable measurement IDs mapped to raw files.
- `processed_ledger.yaml`: processed result records and canonical status.

Create or update a sample:

```powershell
labsuite sample add MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --condition as-grown `
  --replicate R1 `
  --stack "Ta/Co/NiFe/AlOx/NiFe" `
  --geometry-shape square `
  --side 4 `
  --length-unit mm `
  --layer "Ta:2:nm:false" `
  --layer "Co:5:nm:true" `
  --layer "NiFe:2:nm:true" `
  --layer "AlOx:0.5:nm:false" `
  --layer "NiFe:7:nm:true" `
  --estimate-magnetic-volume `
  --save-estimated-volume
```

Register raw files:

```powershell
labsuite sample register-file data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type vsm `
  --sample-id MTJ-NoPattern `
  --measurement-id vsm:MTJ-NoPattern:300K-R1 `
  --notes "300 K VSM loop"

labsuite sample register-file "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type fmr `
  --sample-id MTJ-NoPattern `
  --geometry ip `
  --measurement-id fmr:MTJ-NoPattern:ip-R1

labsuite sample register-file data/raw/esr_sample.dsc `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type esr `
  --sample-id MTJ-NoPattern `
  --geometry angular `
  --measurement-id esr:MTJ-NoPattern:0deg-R1
```

Inspect metadata:

```powershell
labsuite sample list --sample-registry metadata/sample_registry.yaml
labsuite sample show MTJ-NoPattern --sample-registry metadata/sample_registry.yaml
labsuite sample validate --sample-registry metadata/sample_registry.yaml
```

Run analysis and update ledgers:

```powershell
labsuite fmr single `
  --input "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_registry `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --sample-id MTJ-NoPattern `
  --measurement-id fmr:MTJ-NoPattern:ip-R1 `
  --geometry ip `
  --update-ledger `
  --mark-canonical

labsuite vsm batch `
  --input data/raw/VSM/MTJ-NoPattern `
  --pattern "*.dat" `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_registry_batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --metadata-manifest metadata/vsm_manifest.csv `
  --update-ledger `
  --create-sample `
  --mark-canonical `
  --replace-canonical
```

`--update-ledger` records measurement and processed-result metadata. If an input
file is outside `data/raw/`, LabSuite copies it under `data/raw/<MODALITY>/` for
custody and writes `raw_import_map.csv` for batch runs. ESR raw import also
copies the `.DTA` sidecar for `.dsc` files.

For batch metadata manifests, use CSV columns:

```text
raw_path,sample_id,measurement_id,geometry,branch_labels
```

Files that cannot be resolved during registry-aware batch analysis are reported
in `unresolved_files.csv`.

## Higher-Order Sample Analysis

Sample-level analysis consumes canonical processed results from
`metadata/processed_ledger.yaml`. It does not re-parse raw data.

Check readiness:

```powershell
labsuite sample readiness MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml
```

Run one sample:

```powershell
labsuite sample analyze MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived/MTJ-NoPattern
```

Run all registered samples:

```powershell
labsuite sample analyze-batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived
```

Sample analysis writes summary JSON/CSV, report Markdown, provenance snapshots,
sample-level figures, readiness tables, warning tables, and parameter tables.
For FMR measurements with multiple accepted branches, `summary.fmr_branches`
and `tables/fmr_branch_summary.csv` report branch-level `Meff`, `g`, `Ms`, and
`alpha_eff` without averaging magnetic subsystems.

Archive test or trial ledger entries for one sample without deleting provenance:

```powershell
labsuite sample prune-ledger MTJ-NoPattern `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml

labsuite sample prune-ledger MTJ-NoPattern `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --apply
```

The first command is a dry run. `--apply` marks matching measurement and
processed-result records `archived`, so sample analysis ignores them.

## Export and Report Regeneration

Regenerate CSV and figures from a saved analysis JSON:

```powershell
labsuite esr export --input data/processed/esr_single/esr_sample_analysis.json
labsuite fmr export --input data/processed/fmr_single/example_analysis.json --output-dir data/processed/fmr_export
labsuite vsm export --input data/processed/vsm_single/MTJ-NOPATTERN-300K-R1_00001_analysis.json
```

Generate reports:

```powershell
labsuite esr report --input data/processed/esr_single/esr_sample_analysis.json
labsuite fmr report --input data/processed/fmr_batch --recursive
labsuite vsm report --input data/processed/vsm_batch --output data/processed/vsm_batch_report.md
```

If `--input` is a single `*_analysis.json`, the default report path is
`<stem>_report.md`. If `--input` is a result directory, the default report path
is `batch_report.md` inside that directory.

## FMR Polarity Overrides

FMR field-polarity pairing is optional and only meaningful when the experiment
contains positive- and negative-field sweeps for the same sample, frequency,
geometry, and mode.

```powershell
labsuite fmr single `
  --input data/raw/FMR/paired_field_sweeps.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_polarity `
  --field-polarity-correction gonzalez-fuentes `
  --pair-field-polarities `
  --fit-field Hres_avg `
  --require-polarity-pair `
  --on-unpaired-polarity warn_and_keep_raw `
  --polarity-column field_polarity `
  --positive-polarity-labels "positive,pos,+H" `
  --negative-polarity-labels "negative,neg,-H" `
  --pair-by "sample_id,replicate_id,frequency,geometry,mode_id" `
  --max-pair-frequency-tolerance-ghz 0.001 `
  --compare-polarity-fits `
  --plot-polarity-diagnostics
```

Use `--field-polarity-correction none` to explicitly disable this path.

For MTJ-style two-subsystem FMR fits, start with branch tracking and either
float or lock branch-specific `g`:

```powershell
labsuite fmr single `
  --input data/raw/FMR/mtj.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_mtj `
  --n-peaks 2 `
  --enable-branch-tracking `
  --fit-g

labsuite fmr single `
  --input data/raw/FMR/mtj.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_mtj_locked `
  --n-peaks 2 `
  --enable-branch-tracking `
  --branch-lock-g branch_1:2.09 `
  --branch-lock-g branch_2:2.15
```

## Resonance Metrics

ESR and FMR commands compute resonance metrics by default. Options can export
the metrics and add diagnostic figure markers:

```powershell
labsuite esr single `
  --input data/raw/esr_sample.dsc `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_metrics `
  --compute-resonance-metrics `
  --area-window-mode side-aware `
  --area-window-multipliers "1,2,3" `
  --compute-full-area `
  --export-resonance-metrics `
  --plot-halfmax-markers `
  --plot-area-windows

labsuite fmr batch `
  --input data/raw/FMR `
  --pattern "*.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_metrics `
  --no-report-asymmetry `
  --export-resonance-metrics
```

Use `--no-compute-resonance-metrics` to disable the metrics for ESR/FMR runs.

## Output Artifacts

Shared modality artifacts:

- `*_analysis.json`: serialized analysis payload and provenance.
- `*_trace.csv`: point-wise raw, processed, fitted, and diagnostic traces.
- `*_summary.csv`: scalar metrics, fitted parameters, and QC summaries.
- `*_figure.png`: primary per-run figure.
- `analysis_config.yaml`: run configuration snapshot.
- `sample_registry_snapshot.yaml`, `measurement_ledger_snapshot.yaml`,
  `processed_ledger_snapshot.yaml`: metadata snapshots.
- `batch_summary.csv`: one row per discovered source.
- `batch_manifest.json`: machine-readable batch inventory.
- `unresolved_files.csv`: unresolved registry-aware batch inputs, when present.
- `raw_import_map.csv`: copied raw-file custody map, when present.

ESR batch artifacts include `batch_qc.csv`, angle overlays, and processed
offset overlays. FMR outputs include assembled series CSV files and optional
trace diagnostics. VSM outputs include hysteresis overlays.

## Modality Notes

ESR parses Bruker `.dsc` descriptors plus sibling `.DTA` data, applies explicit
baseline and smoothing choices from the recipe, integrates derivative traces,
fits derivative Lorentzians, and exports batch QC when run in batch mode.

FMR parses PhaseFMR `.log` files, fits single or double mixed derivative
Lorentzian trace models, assembles accepted resonance series, and fits Kittel
and linewidth trends when enough accepted points are available.

VSM parses `.dat` loop exports, applies explicit preprocessing, splits loop
branches, evaluates high-field background subtraction, extracts hysteresis
observables, and records trust and uncertainty diagnostics.
