# Metadata

This folder contains the project-level metadata stores used to make LabSuite
analysis reproducible across raw imports, processed results, and sample-level
derived analysis.

Run examples from the `labsuite/` package directory:

```powershell
cd F:\LabAnalysisSuite\labsuite
```

Default paths:

```text
metadata/sample_registry.yaml
metadata/measurement_ledger.yaml
metadata/processed_ledger.yaml
```

## Metadata Files

`sample_registry.yaml` records physical samples. It is keyed by stable
`sample_id` values and stores aliases, condition, replicate, stack notes,
geometry, layer stack, magnetic volume metadata, and analysis defaults such as
FMR `g_mode`.

`measurement_ledger.yaml` records raw measurement files. It maps stable
`measurement_id` values to `sample_id`, modality, raw path, geometry, branch
labels, instrument notes, and free-text notes. This keeps raw measurement
identity independent of filenames.

`processed_ledger.yaml` records processed analysis results. It maps stable
`result_id` values to a measurement, sample, modality, processed JSON path,
recipe path and hash, creation time, canonical status, and summary metadata.
Sample-level analysis consumes canonical processed results from this ledger.

## Sample Registry Shape

The registry is keyed by `sample_id` under `samples`.

```yaml
schema_version: 2
samples:
  MTJ-NoPattern:
    sample_id: MTJ-NoPattern
    aliases: []
    condition: as-grown
    replicate: R1
    stack: Ta/Co/NiFe/AlOx/NiFe
    geometry:
      area:
        value: null
        unit: null
        uncertainty: null
      magnetic_thickness:
        value: null
        unit: null
        uncertainty: null
      vmag:
        value: null
        unit: null
        uncertainty: null
        method: null
      shape: square
      dimensions:
        side: 4
        side_unit: mm
    layer_stack:
      - material: Ta
        thickness: 2
        thickness_unit: nm
        magnetic: false
      - material: Co
        thickness: 5
        thickness_unit: nm
        magnetic: true
    magnetic_volume_m3: 8.0e-14
    magnetic_volume_source: estimated
    magnetic_volume_method: geometry_area_times_magnetic_layer_thickness
    defaults:
      g_mode: float
      g_value: null
      ms_source: null
```

Important fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Registry file format version. Current value is `2`. |
| `sample_id` | Stable physical sample identifier. Prefer this over filename-derived IDs. |
| `aliases` | Alternate names that can resolve to the sample. |
| `condition` | Sample treatment, state, temperature condition, or operator label. |
| `replicate` | Physical replicate label such as `R1`. |
| `stack` | Human-readable stack description. |
| `geometry.area` | Area value, unit, and uncertainty. |
| `geometry.magnetic_thickness` | Magnetic thickness value, unit, and uncertainty. |
| `geometry.vmag` | Direct magnetic volume value, unit, uncertainty, and method. |
| `geometry.shape` | Shape used for estimated magnetic volume: `rectangle`, `square`, `circle`, `custom_area`, or `array`. |
| `layer_stack` | Ordered layer list. Only entries with `magnetic: true` contribute to estimated magnetic volume. |
| `magnetic_volume_m3` | Canonical magnetic volume in cubic meters. Manual values override estimates. |
| `defaults.g_mode` | FMR gamma/g handling: `float`, `fixed`, or `bounded`. |
| `defaults.g_value` | Numeric g value used for fixed or bounded modes. |
| `defaults.ms_source` | Reference to an Ms source, usually a VSM measurement or note. |

## Ledger Shapes

Measurement ledger:

```yaml
schema_version: 1
measurements:
  fmr:MTJ-NoPattern:ip-R1:
    measurement_id: fmr:MTJ-NoPattern:ip-R1
    sample_id: MTJ-NoPattern
    type: fmr
    raw_path: F:\LabAnalysisSuite\labsuite\data\raw\FMR\example.log
    geometry: ip
    branch_labels: []
    instrument: null
    notes: null
```

Processed ledger:

```yaml
schema_version: 1
processed_results:
  fmr_MTJ-NoPattern_example_default_2026-04-28T22_07_31Z:
    result_id: fmr_MTJ-NoPattern_example_default_2026-04-28T22_07_31Z
    measurement_id: fmr:MTJ-NoPattern:ip-R1
    sample_id: MTJ-NoPattern
    type: fmr
    processed_path: F:\LabAnalysisSuite\labsuite\data\processed\fmr_registry\example_analysis.json
    recipe_path: F:\LabAnalysisSuite\labsuite\recipes\fmr\default.yaml
    recipe_hash: <sha256>
    created_at: '2026-04-28T22:07:31Z'
    status: canonical
    summary:
      geometry: ip
      g_mode: float
      g_value: null
      branches:
        - mode_1
```

Use one canonical processed result per sample, modality, measurement, recipe,
and branch context unless you intentionally replace it with
`--replace-canonical`.

## Sample Commands

Add or update sample metadata:

```powershell
labsuite sample add MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --alias MTJ-NoPattern-old `
  --condition as-grown `
  --replicate R1 `
  --stack "Ta/Co/NiFe/AlOx/NiFe" `
  --area-value 2.0 `
  --area-unit mm^2 `
  --area-uncertainty 0.1 `
  --thickness-value 1.5 `
  --thickness-unit nm `
  --thickness-uncertainty 0.05 `
  --g-mode bounded `
  --g-value 2.10 `
  --ms-source vsm:MTJ-NoPattern:300K-R1

labsuite sample update MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --condition annealed
```

Store a manual canonical magnetic volume:

```powershell
labsuite sample update MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --magnetic-volume-m3 1.23e-13
```

Estimate magnetic volume from geometry and magnetic layers:

```powershell
labsuite sample update MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
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

Inspect and validate:

```powershell
labsuite sample list --sample-registry metadata/sample_registry.yaml
labsuite sample show MTJ-NoPattern --sample-registry metadata/sample_registry.yaml
labsuite sample validate --sample-registry metadata/sample_registry.yaml
```

## Register Raw Files

Register a VSM file:

```powershell
labsuite sample register-file data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type vsm `
  --sample-id MTJ-NoPattern `
  --measurement-id vsm:MTJ-NoPattern:300K-R1 `
  --notes "300 K VSM loop"
```

Register an FMR file:

```powershell
labsuite sample register-file "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type fmr `
  --sample-id MTJ-NoPattern `
  --geometry ip `
  --measurement-id fmr:MTJ-NoPattern:ip-R1
```

Register an ESR file:

```powershell
labsuite sample register-file data/raw/esr_sample.dsc `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type esr `
  --sample-id MTJ-NoPattern `
  --geometry angular `
  --measurement-id esr:MTJ-NoPattern:0deg-R1
```

Interactive registration is available when an operator needs prompts:

```powershell
labsuite sample add --sample-registry metadata/sample_registry.yaml --interactive

labsuite sample register-file data/raw/esr_sample.dsc `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --type esr `
  --interactive
```

## Ledger-Producing Workflow

Run modality analysis with `--update-ledger` to update both measurement and
processed ledgers. Add `--mark-canonical` when the result should be consumed by
sample-level analysis.

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
```

For batch runs, use a metadata manifest when many files need sample and
measurement IDs:

```csv
raw_path,sample_id,measurement_id,geometry,branch_labels
data/raw/VSM/MTJ-NoPattern/MTJ-NOPATTERN-300K-R1_00001.dat,MTJ-NoPattern,vsm:MTJ-NoPattern:300K-R1,unknown,
```

```powershell
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

If a registry-aware batch cannot resolve a file, LabSuite writes
`unresolved_files.csv` in the batch output directory. If raw files are imported
into `data/raw/` during ledger update, batch runs write `raw_import_map.csv`.

Each registry-aware result directory contains:

```text
analysis_config.yaml
sample_registry_snapshot.yaml
measurement_ledger_snapshot.yaml
processed_ledger_snapshot.yaml
```

The analysis JSON also records resolved metadata in provenance so results remain
reproducible if the live metadata files change later.

## Higher-Order Analysis

Sample-level derived analysis consumes canonical processed results from
`processed_ledger.yaml`.

```powershell
labsuite sample readiness MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml

labsuite sample analyze MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived/MTJ-NoPattern

labsuite sample analyze-batch `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --recipe recipes/sample_analysis/default.yaml `
  --output-dir data/derived
```

## Analysis Overrides

Analysis commands can override registry defaults for one run:

```powershell
labsuite fmr single `
  --input "data/raw/FMR/Temp2-Co-A-2,5to17GHz-R1 - 20260226-130701.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_g_fixed `
  --sample-registry metadata/sample_registry.yaml `
  --measurement-ledger metadata/measurement_ledger.yaml `
  --processed-ledger metadata/processed_ledger.yaml `
  --sample-id MTJ-NoPattern `
  --geometry ip `
  --g-mode fixed `
  --g-value 2.10
```

`g_mode` behavior:

| Mode | Behavior |
| --- | --- |
| `float` | Fit gamma/g normally. |
| `fixed` | Use `g_value` to fix gamma during the Kittel fit. |
| `bounded` | Use `g_value` to constrain gamma around that value. |

## Scientific Notes

- `magnetic_volume_m3` is the canonical registry magnetic volume in SI cubic
  meters.
- Manual magnetic volume values override estimated values.
- Estimated magnetic volume uses only `layer_stack` entries marked
  `magnetic: true`.
- VSM geometry metadata is provenance only. VSM analysis does not silently
  normalize moment by area, thickness, or volume.
- FMR analysis does not require VSM metadata. If Ms is unavailable, FMR still
  calculates `M_eff` and skips anisotropy K calculations with a warning.
- Parser output remains raw imported data. Registry metadata is applied at the
  workflow/provenance layer, not as hidden parser cleanup.

## Validation Warnings

`labsuite sample validate` reports warnings for missing units, missing FMR/ESR
geometry, duplicate aliases, missing measurement paths, incomplete direct volume
metadata, missing volume inputs, non-positive or unrecognized magnetic volume
metadata, and resolver warnings from geometry or layer-stack estimates.

Warnings are non-fatal. Malformed YAML and duplicate sample IDs are errors.
