# Recipes

Recipes are YAML configuration files that make preprocessing, fitting, QC, and
sample-level derived analysis explicit. They are part of the reproducibility
record: analysis outputs preserve recipe snapshots so a run can be reconstructed
later.

Run examples from the `labsuite/` package directory:

```powershell
cd F:\LabAnalysisSuite\labsuite
```

## Current Recipes

- `recipes/esr/default.yaml`: derivative baseline handling, smoothing,
  resonance-window detection, single/split derivative-Lorentzian fitting, local
  integration, and ESR batch QC thresholds.
- `recipes/fmr/default.yaml`: signal channel selection, edge baseline
  subtraction, optional smoothing, single/double mixed derivative Lorentzian
  fitting, component QC, Kittel fitting, linewidth fitting, and optional
  positive/negative field-polarity pairing.
- `recipes/vsm/default.yaml`: loop preprocessing, high-field tail selection,
  background correction acceptance rules, centering, hysteresis metrics, trust
  diagnostics, and uncertainty settings.
- `recipes/sample_analysis/default.yaml`: higher-order sample analysis across
  canonical processed VSM, FMR, and ESR results, including readiness checks,
  magnetization policy, anisotropy policy, damping policy, warnings, and export
  controls.

## Editing Workflow

Start from a default recipe and copy it before tuning:

```powershell
Copy-Item recipes/fmr/default.yaml recipes/fmr/MTJ-NoPattern_ip.yaml
```

Run with the copied recipe:

```powershell
labsuite fmr single `
  --input "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --recipe recipes/fmr/MTJ-NoPattern_ip.yaml `
  --output-dir data/processed/fmr_MTJ-NoPattern_ip
```

Print or write defaults from the CLI:

```powershell
labsuite esr config
labsuite fmr config --output recipes/fmr/from_cli_default.yaml
labsuite vsm config --output recipes/vsm/from_cli_default.yaml
```

Treat recipe edits as scientific decisions. Do not encode hidden corrections in
parsers or GUI code. If a baseline, smoothing, normalization, field-polarity
pairing, fit constraint, or QC threshold changes the result, it belongs in a
recipe or explicit analysis option.

## Proposed Analysis Workflow

1. Register sample metadata in `metadata/sample_registry.yaml`.
2. Register raw measurements, or run analysis with enough sample metadata to
   update ledgers.
3. Run modality-level processing into `data/processed/` with an explicit
   modality recipe.
4. Mark the chosen processed results canonical in `metadata/processed_ledger.yaml`.
5. Run `labsuite sample readiness` to check whether the sample has enough
   canonical processed inputs for derived analysis.
6. Run `labsuite sample analyze` for one sample or `labsuite sample analyze-batch`
   for all registered samples.

## Modality Recipe Commands

ESR:

```powershell
labsuite esr single `
  --input data/raw/esr_sample.dsc `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_single `
  --fit-mode auto

labsuite esr batch `
  --input data/raw/ESR `
  --pattern "*.dsc" `
  --recursive `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_batch
```

FMR:

```powershell
labsuite fmr single `
  --input "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_single

labsuite fmr batch `
  --input data/raw/FMR `
  --pattern "*.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_batch
```

VSM:

```powershell
labsuite vsm single `
  --input data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_single

labsuite vsm batch `
  --input data/raw/VSM/MTJ-NoPattern `
  --pattern "*.dat" `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_batch
```

## Sample-Level Recipe Commands

Check readiness:

```powershell
labsuite sample readiness MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml
```

Analyze one sample:

```powershell
labsuite sample analyze MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived/MTJ-NoPattern
```

Analyze all registered samples:

```powershell
labsuite sample analyze-batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived
```

## Recipe Provenance

Modality runs write `analysis_config.yaml` and metadata snapshots into their
output directories. Sample-level analysis writes
`provenance/analysis_recipe_snapshot.yaml` and
`provenance/processed_inputs_manifest.json`. Keep those files with exported
tables and figures when archiving results.
