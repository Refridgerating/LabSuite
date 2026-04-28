# Sample Registry

`sample_registry.yaml` is the project-level metadata registry for physical
samples. It links stable sample IDs to VSM, FMR, and ESR measurement files so
analysis does not depend only on filenames.

Default registry path:

```powershell
metadata/sample_registry.yaml
```

Run the examples from the `labsuite` project directory:

```powershell
cd F:\LabAnalysisSuite\labsuite
```

## YAML Shape

The registry is keyed by `sample_id` under `samples`.

```yaml
schema_version: 2
samples:
  MTJ-NoPattern:
    sample_id: MTJ-NoPattern
    aliases: []
    condition: null
    replicate: null
    stack: null
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
      shape: rectangle
      dimensions:
        length: 5
        width: 5
        unit: mm
      notes: null
    layer_stack:
      - material: Ta
        thickness: 2
        thickness_unit: nm
        magnetic: false
      - material: Co
        thickness: 5
        thickness_unit: nm
        magnetic: true
    magnetic_volume_m3: null
    magnetic_volume_source: unknown
    magnetic_volume_method: null
    magnetic_volume_warnings: []
    defaults:
      g_mode: float
      g_value: null
      ms_source: null
    measurements:
      - measurement_id: fmr:example
        sample_id: MTJ-NoPattern
        type: fmr
        path: data/raw/FMR/example.log
        geometry: ip
        branch_labels: []
        notes: null
```

## Field Reference

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | Yes | Registry file format version. Current value is `2`. |
| `samples` | Yes | Mapping of sample IDs to sample records. |
| `sample_id` | Yes | Stable physical sample identifier. Prefer this over filename-derived IDs. |
| `aliases` | No | Alternate names that can resolve to the sample, such as old filename stems. |
| `condition` | No | Sample condition label, treatment, temperature state, or other operator note. |
| `replicate` | No | Physical replicate label such as `R1`. |
| `stack` | No | Stack description or layer sequence. |
| `geometry.area.value` | No | Sample area numeric value. |
| `geometry.area.unit` | Required when value is set | Area unit, for example `mm^2` or `cm^2`. |
| `geometry.area.uncertainty` | No | Area uncertainty in the same unit. |
| `geometry.magnetic_thickness.value` | No | Magnetic thickness numeric value. |
| `geometry.magnetic_thickness.unit` | Required when value is set | Thickness unit, for example `nm`. |
| `geometry.magnetic_thickness.uncertainty` | No | Thickness uncertainty in the same unit. |
| `geometry.vmag.value` | No | Direct magnetic volume value. |
| `geometry.vmag.unit` | Required when value is set | Direct volume unit, for example `m^3` or `cm^3`. |
| `geometry.vmag.uncertainty` | No | Direct volume uncertainty in the same unit. |
| `geometry.vmag.method` | Required for direct `vmag` | How direct volume was measured or calculated. |
| `geometry.shape` | No | Shape used for registry-level magnetic-volume estimates: `rectangle`, `square`, `circle`, `custom_area`, or `array`. |
| `geometry.dimensions` | Required for estimates | Shape dimensions and units used by the shared magnetic-volume estimator. |
| `layer_stack` | No | Ordered physical layers. Only layers with `magnetic: true` contribute to estimated magnetic volume. |
| `magnetic_volume_m3` | No | Canonical SI magnetic volume in cubic meters. Manual values override estimates. |
| `magnetic_volume_source` | No | `manual`, `estimated`, `imported`, or `unknown`. |
| `magnetic_volume_method` | No | Provenance method such as `manual` or `geometry_area_times_magnetic_layer_thickness`. |
| `magnetic_volume_warnings` | No | Warnings emitted by the resolver or estimator. |
| `defaults.g_mode` | No | FMR gamma/g handling: `float`, `fixed`, or `bounded`. Defaults to `float`. |
| `defaults.g_value` | Required for useful `fixed` or `bounded` g modes | Numeric g value used to fix or bound gamma. |
| `defaults.ms_source` | No | Reference to the Ms source, such as a VSM measurement ID or external note. |
| `measurements` | No | List of files associated with this sample. |
| `measurement_id` | Yes | Stable measurement ID, usually `<type>:<stem>`. |
| `measurement.sample_id` | Yes | Sample ID this measurement belongs to. |
| `measurement.type` | Yes | One of `vsm`, `fmr`, or `esr`. |
| `measurement.path` | Yes | File path. Relative paths are resolved from the registry folder. |
| `measurement.geometry` | Required for FMR/ESR | One of `ip`, `oop`, `angular`, or `unknown`. |
| `measurement.branch_labels` | No | Optional labels for known branches/modes. |
| `measurement.notes` | No | Free text measurement note. |

## Non-Interactive Workflow

Non-interactive commands never prompt. If single-file analysis cannot resolve
the sample/file from the registry, it fails cleanly. Batch analysis skips
unresolved files and writes them to `unresolved_files.csv`.

Add or update sample metadata:

```powershell
labsuite sample add MTJ-NoPattern `
  --registry metadata/sample_registry.yaml `
  --alias MTJ-NoPattern-old `
  --condition as-grown `
  --replicate R1 `
  --stack "Ta/Pt/CoFeB/MgO" `
  --area-value 2.0 `
  --area-unit mm^2 `
  --area-uncertainty 0.1 `
  --thickness-value 1.5 `
  --thickness-unit nm `
  --thickness-uncertainty 0.05 `
  --g-mode bounded `
  --g-value 2.10 `
  --ms-source vsm:MTJ-NoPattern-300K
```

Store a manual canonical magnetic volume:

```powershell
labsuite sample update MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --magnetic-volume-m3 1.23e-13
```

Estimate magnetic volume from rectangle geometry and magnetic layers:

```powershell
labsuite sample update MTJ-NoPattern `
  --sample-registry metadata/sample_registry.yaml `
  --geometry-shape rectangle `
  --length 5 --width 5 --length-unit mm `
  --layer "Ta:2:nm:false" `
  --layer "Co:5:nm:true" `
  --layer "NiFe:2:nm:true" `
  --layer "AlOx:0.5:nm:false" `
  --layer "NiFe:7:nm:true" `
  --estimate-magnetic-volume `
  --save-estimated-volume
```

Register a VSM file:

```powershell
labsuite sample register-file data/raw/MTJ-NoPattern-300K-R1_00001.dat `
  --registry metadata/sample_registry.yaml `
  --type vsm `
  --sample-id MTJ-NoPattern `
  --measurement-id vsm:MTJ-NoPattern-300K `
  --notes "300 K VSM loop"
```

Register an FMR file:

```powershell
labsuite sample register-file data/raw/FMR/MTJ-NoPattern-2to18GHz-R1.log `
  --registry metadata/sample_registry.yaml `
  --type fmr `
  --sample-id MTJ-NoPattern `
  --geometry ip `
  --measurement-id fmr:MTJ-NoPattern-ip-R1
```

Register an ESR file:

```powershell
labsuite sample register-file data/raw/ESR/MTJ-NoPattern-0deg-R1.dsc `
  --registry metadata/sample_registry.yaml `
  --type esr `
  --sample-id MTJ-NoPattern `
  --geometry angular `
  --measurement-id esr:MTJ-NoPattern-0deg-R1
```

Inspect and validate:

```powershell
labsuite sample list --registry metadata/sample_registry.yaml
labsuite sample show MTJ-NoPattern --registry metadata/sample_registry.yaml
labsuite sample validate --registry metadata/sample_registry.yaml
```

Run single-file analysis with the registry:

```powershell
labsuite fmr single `
  --input data/raw/FMR/MTJ-NoPattern-2to18GHz-R1.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_registry_test `
  --registry metadata/sample_registry.yaml

labsuite vsm single `
  --input data/raw/MTJ-NoPattern-300K-R1_00001.dat `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_registry_test `
  --registry metadata/sample_registry.yaml

labsuite esr single `
  --input data/raw/ESR/MTJ-NoPattern-0deg-R1.dsc `
  --recipe recipes/esr/default.yaml `
  --output-dir data/processed/esr_registry_test `
  --registry metadata/sample_registry.yaml
```

Run batch analysis with unresolved-file reporting:

```powershell
labsuite fmr batch `
  --input data/raw/FMR `
  --pattern "*.log" `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_registry_batch `
  --registry metadata/sample_registry.yaml

labsuite vsm batch `
  --input data/raw `
  --pattern "*.dat" `
  --recipe recipes/vsm/default.yaml `
  --output-dir data/processed/vsm_registry_batch `
  --registry metadata/sample_registry.yaml
```

If a batch contains files not registered to any sample, check:

```powershell
data/processed/<batch_output>/unresolved_files.csv
```

## Interactive Workflow

Interactive commands may prompt for missing sample identity. Interactive
analysis can register an unregistered file before running the analysis.

Add a sample interactively:

```powershell
labsuite sample add --registry metadata/sample_registry.yaml --interactive
```

Register a file interactively:

```powershell
labsuite sample register-file data/raw/FMR/MTJ-NoPattern-2to18GHz-R1.log `
  --registry metadata/sample_registry.yaml `
  --type fmr `
  --geometry ip `
  --interactive
```

Run analysis interactively:

```powershell
labsuite fmr single `
  --input data/raw/FMR/MTJ-NoPattern-2to18GHz-R1.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_registry_interactive `
  --registry metadata/sample_registry.yaml `
  --interactive
```

The interactive analysis command prompts for a sample ID if the file is not
already registered. If the sample does not exist, the CLI creates a minimal
sample entry and registers the file.

## Analysis Overrides

Analysis commands can override registry defaults for one run:

```powershell
labsuite fmr single `
  --input data/raw/FMR/MTJ-NoPattern-2to18GHz-R1.log `
  --recipe recipes/fmr/default.yaml `
  --output-dir data/processed/fmr_g_fixed `
  --registry metadata/sample_registry.yaml `
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
| `bounded` | Use `g_value` to constrain gamma within the implemented bound around that value. |

## Scientific Notes

- `magnetic_volume_m3` is the canonical registry magnetic volume in SI cubic
  meters.
- Manual magnetic volume values override estimated values.
- Estimated magnetic volume uses only `layer_stack` entries marked
  `magnetic: true`; nonmagnetic layers are retained as provenance but excluded
  from the thickness sum.
- VSM geometry metadata is provenance only. The VSM analysis does not silently
  normalize moment by area, thickness, or volume.
- A later VSM pass will use registry magnetic volume to compute `Ms_A_per_m`
  while keeping existing `Ms_emu` outputs unchanged.
- FMR analysis does not require VSM metadata. If Ms is unavailable, FMR still
  calculates `M_eff` and skips anisotropy K calculations with a warning.
- Parser output remains raw imported data. Registry metadata is applied at the
  workflow/provenance layer, not as hidden parser cleanup.

## Result Provenance

Each registry-aware result directory contains:

```text
analysis_config.yaml
sample_registry_snapshot.yaml
```

The analysis JSON also records resolved registry metadata in provenance. This
keeps the run reproducible even if `metadata/sample_registry.yaml` changes later.

## Validation Warnings

`labsuite sample validate` reports warnings for:

- missing units on area, magnetic thickness, or direct volume values
- missing FMR/ESR geometry
- duplicate aliases across samples
- measurement paths that do not exist
- incomplete direct `vmag` metadata
- missing volume inputs when neither direct `vmag` nor area plus thickness is complete
- non-positive or unrecognized registry-level magnetic volume metadata
- resolver warnings from geometry/layer-stack estimates

Warnings are non-fatal. Malformed YAML or duplicate sample IDs are treated as
errors.
