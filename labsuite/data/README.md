# Data Folder

This folder is the project data workspace. The same boundary used in the code
applies here:

```text
raw import -> explicit preprocessing/fitting -> processed result -> derived sample analysis
```

Do not use this folder structure to hide scientific cleanup. Raw files stay raw.
Preprocessing, fitting, and derived calculations must be represented by recipes,
analysis JSON, ledgers, and provenance snapshots.

## Folder Roles

`data/raw/` contains immutable instrument exports and required sidecars. Use it
for original ESR, FMR, and VSM files. ESR `.dsc` files need their sibling `.DTA`
data file. Registry-aware analysis with `--update-ledger` can copy external raw
files into `data/raw/<MODALITY>/` for project custody.

`data/intermediate/` is for explicit temporary or normalized workflow products
that are not final scientific results. Anything placed here should be
regenerable from raw data plus recipes or scripts. Do not treat intermediate
files as canonical outputs.

`data/processed/` contains modality-level analysis outputs. Single-file and
batch CLI runs write analysis JSON, trace CSVs, summary CSVs, figures, batch
manifests, reports, and metadata snapshots here by default.

`data/derived/` is the default destination for higher-order sample-level
analysis from `labsuite sample analyze` and `labsuite sample analyze-batch`.
Those workflows consume canonical processed results from
`metadata/processed_ledger.yaml`; they do not re-parse raw data.

## Raw vs Processed vs Intermediate

Raw data is the instrument output plus sidecars and import metadata. Parsers may
read files, detect format, map columns, preserve metadata, and return structured
raw data. Parsers must not smooth, normalize, subtract baselines, silently drop
noisy points, or apply publication corrections.

Intermediate data is explicit work product between raw import and final
analysis. Use it for temporary conversions, cached tables, or normalized staging
only when the generating step is documented and reproducible.

Processed data is the output of a recipe-driven analysis. It includes the
preprocessing recipe, fit recipe, fitted parameters, derived parameters, export
settings, software context where available, and provenance snapshots.

Derived data is sample-level analysis across processed VSM, FMR, and ESR
results. It summarizes physical parameters and quality warnings from canonical
processed inputs.

## Expected Artifacts

Modality single-file outputs commonly include:

```text
*_analysis.json
*_trace.csv
*_summary.csv
*_figure.png
analysis_config.yaml
sample_registry_snapshot.yaml
measurement_ledger_snapshot.yaml
processed_ledger_snapshot.yaml
```

Batch outputs commonly include:

```text
batch_summary.csv
batch_manifest.json
unresolved_files.csv
raw_import_map.csv
```

`unresolved_files.csv` appears when registry-aware batch analysis cannot resolve
all inputs. `raw_import_map.csv` appears when batch ledger updates import raw
files into the project raw tree.

Modality-specific batch artifacts can include:

- ESR: `batch_qc.csv`, angle overlays, and processed offset overlays.
- FMR: `*_series.csv` files and trace diagnostic figures.
- VSM: `batch_hysteresis_overlay.png`.

Sample-level derived outputs commonly include:

```text
sample_analysis_summary.json
sample_analysis_summary.csv
sample_analysis_report.md
tables/readiness_matrix.csv
tables/fmr_branch_parameters.csv
tables/vsm_parameters.csv
tables/esr_parameters.csv
tables/anisotropy_parameters.csv
tables/damping_parameters.csv
warnings/analysis_warnings.csv
provenance/sample_registry_snapshot.yaml
provenance/measurement_ledger_snapshot.yaml
provenance/processed_ledger_snapshot.yaml
provenance/analysis_recipe_snapshot.yaml
provenance/processed_inputs_manifest.json
figures/sample_analysis_overview.png
```

## Common Commands

Run raw single-file analysis into `data/processed/`:

```powershell
labsuite vsm single `
  --input data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_single
```

Run registry-aware batch analysis and update ledgers:

```powershell
labsuite fmr batch `
  --input data/raw/FMR `
  --pattern "*.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --metadata-manifest metadata/fmr_manifest.csv `
  --update-ledger `
  --mark-canonical
```

Run higher-order sample analysis into `data/derived/`:

```powershell
labsuite sample analyze-batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived
```
