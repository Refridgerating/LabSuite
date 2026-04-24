# LabSuite

LabSuite is a reproducible scientific analysis platform for ESR, FMR, and VSM measurements. The current operator surface is the shared `labsuite` CLI plus modality-specific backend plugins under `src/labsuite/plugins/`.

This repository-level README is repo-facing. The installable Python package lives in `labsuite/`, so command examples below assume you first change into that directory.

## Shared Pipeline

Every implemented workflow follows the same contract:

`parse -> preprocess -> fit -> derive -> export`

The separation is strict:

- Parsing reads raw files and preserves metadata.
- Preprocessing performs explicit, recipe-driven cleanup or correction.
- Fitting applies the selected scientific model.
- Derived parameters are computed after fitting.
- Export writes reproducible artifacts to disk.

This is the main architectural rule behind the codebase: thin GUI, shared backend, modality-specific science in plugins, and no hidden scientific state inside widgets.

## Setup

LabSuite currently targets Python 3.11+.

From the repository root:

```bash
cd labsuite
python -m pip install -e .
```

For development extras:

```bash
cd labsuite
python -m pip install -e ".[dev]"
```

All path examples below assume your current working directory is `labsuite/`. If you stay at the repo root, prefix file paths with `labsuite/`.

## Shared vs Modality-Specific Code

Shared layers:

- `src/labsuite/core/`: shared schemas, preprocessing primitives, fitting abstractions, export helpers, reporting, and batch infrastructure
- `src/labsuite/io/`: shared file sniffing and generic parser helpers
- `src/labsuite/workflows/`: single-file and batch orchestration used by the CLI and future GUI
- `src/labsuite/cli/`: command surface for single, batch, export, config, and report workflows

Modality-specific layers:

- `src/labsuite/plugins/esr/`: Bruker ESR parsing, derivative-Lorentzian fitting, local integration, ESR batch QC, ESR reports and overlays
- `src/labsuite/plugins/fmr/`: PhaseFMR log parsing, trace fitting, resonance-series assembly, Kittel and linewidth fits, FMR reports
- `src/labsuite/plugins/vsm/`: VSM loop parsing, branch splitting, high-field background analysis, hysteresis metrics, uncertainty and trust diagnostics

Presentation layer:

- `src/labsuite/gui/`: Qt models, widgets, dialogs, and plots

Important rule:

- GUI code triggers services and renders results. Physics equations, fitting logic, preprocessing rules, and derived parameters do not belong in widgets or dialogs.

Not yet operator-facing:

- `src/labsuite/workflows/batch_manifest.py`
- `src/labsuite/workflows/publication_export.py`
- empty CLI/app scaffolding modules

Those files are placeholders, not documented runtime features.

## Current CLI Surface

Primary entrypoint:

```bash
labsuite --version
```

Current modality verbs:

```text
labsuite esr single|batch|config|export|report
labsuite vsm single|batch|config|export|report
labsuite fmr single|batch|config|export|report
labsuite fit-single
labsuite fit-batch
labsuite esr-single
labsuite-gui
```

Notes:

- `labsuite fit-single` and `labsuite fit-batch` are ESR-only compatibility commands.
- `labsuite esr-single` is the legacy ESR single-file alias.
- `labsuite-gui` is declared in package metadata, but `src/labsuite/app/` is still scaffold-only and not ready as an operator workflow.

### Single-file Commands

ESR:

```bash
labsuite esr single --input PATH --recipe PATH --output-dir DIR [--fit-mode auto|single|split] [--show-raw]
```

VSM:

```bash
labsuite vsm single --input PATH --recipe PATH --output-dir DIR
```

FMR:

```bash
labsuite fmr single --input PATH --recipe PATH --output-dir DIR
```

Behavior:

- `--input` can be a direct source file or a directory that resolves to exactly one discoverable source.
- If `--output-dir` is omitted, LabSuite writes into `data/processed/<source-stem>/`.
- Default discovery suffixes are `.dsc` for ESR, `.dat` for VSM, and `.log` for FMR.

### Batch Commands

ESR:

```bash
labsuite esr batch --input PATH --pattern "*.dsc" [--recursive] --recipe PATH --output-dir DIR [--fit-mode auto|single|split]
```

VSM:

```bash
labsuite vsm batch --input PATH --pattern "*.dat" [--recursive] --recipe PATH --output-dir DIR
```

FMR:

```bash
labsuite fmr batch --input PATH --pattern "*.log" [--recursive] --recipe PATH --output-dir DIR
```

Behavior:

- `--input` can be a file or folder.
- `--pattern` filters discovered source files.
- `--recursive` switches folder scanning from `glob()` to recursive discovery.
- If `--output-dir` is omitted, batch runs default to `data/processed/<input-name>_<timestamp>/`.

### Config, Export, and Report

Print or write the default recipe for a modality:

```bash
labsuite esr config [--output PATH]
labsuite vsm config [--output PATH]
labsuite fmr config [--output PATH]
```

Regenerate exports from saved analysis JSON:

```bash
labsuite esr export --input PATH_TO_ANALYSIS_JSON [--output-dir DIR]
labsuite vsm export --input PATH_TO_ANALYSIS_JSON [--output-dir DIR]
labsuite fmr export --input PATH_TO_ANALYSIS_JSON [--output-dir DIR]
```

Generate Markdown reports from one JSON file or a result directory:

```bash
labsuite esr report --input PATH [--output PATH] [--recursive]
labsuite vsm report --input PATH [--output PATH] [--recursive]
labsuite fmr report --input PATH [--output PATH] [--recursive]
```

Report behavior:

- If `--input` is a single `*_analysis.json`, the default output is `<stem>_report.md`.
- If `--input` is a result directory, the default output is `batch_report.md` inside that directory.

### ESR Compatibility Commands

```bash
labsuite fit-single --input PATH --recipe PATH --output-dir DIR [--fit-mode auto|single|split] [--show-raw]
labsuite fit-batch --input PATH --pattern "*.dsc" [--recursive] --recipe PATH --output-dir DIR [--fit-mode auto|single|split]
labsuite esr-single SOURCE_FILE [--recipe PATH] [--output-dir DIR] [--fit-mode auto|single|split] [--show-raw]
```

These commands exist for compatibility. Prefer `labsuite esr single` and `labsuite esr batch` for new usage.

## Example Commands

Print the default ESR recipe:

```bash
labsuite esr config
```

Run one ESR analysis:

```bash
labsuite esr single \
  --input data/raw/esr_sample.dsc \
  --recipe recipes/esr/default.yaml \
  --output-dir data/processed/esr_single \
  --fit-mode auto
```

Run an ESR batch:

```bash
labsuite esr batch \
  --input data/raw \
  --pattern "*.dsc" \
  --recipe recipes/esr/default.yaml \
  --output-dir data/processed/esr_batch
```

Run one FMR analysis:

```bash
labsuite fmr single \
  --input data/raw/Temp2-Co-A-2,5to17GHz-R1.log \
  --recipe recipes/fmr/default.yaml \
  --output-dir data/processed/fmr_single
```

Run a VSM batch:

```bash
labsuite vsm batch \
  --input data/raw \
  --pattern "MTJ-B-*.dat" \
  --recipe recipes/vsm/default.yaml \
  --output-dir data/processed/vsm_batch
```

Regenerate artifacts from saved JSON:

```bash
labsuite fmr export \
  --input data/processed/fmr_single/Temp2-Co-A-2,5to17GHz-R1_analysis.json \
  --output-dir data/processed/fmr_export
```

Build a batch report from an existing results directory:

```bash
labsuite vsm report \
  --input data/processed/vsm_batch \
  --output data/processed/vsm_batch_report.md
```

## Output Artifacts

Shared artifacts:

- `*_analysis.json`: full serialized analysis payload and provenance
- `*_trace.csv`: point-wise export for raw, processed, fitted, and diagnostic traces
- `*_summary.csv`: scalar metrics, parameters, QC, and recipe-dependent summaries
- `*_figure.png`: primary per-run figure
- `batch_summary.csv`: one row per discovered source in a batch run
- `batch_manifest.json`: machine-readable batch inventory, status, and batch figures

ESR-only artifacts:

- `batch_qc.csv`: batch QC table with acceptance, duplicate selection, and ESR-specific quality metrics
- `batch_angle_overlay_<replicate>.png`: accepted traces overlaid by nominal angle
- `batch_processed_offset_<replicate>.png`: accepted processed derivative traces stacked with offsets

FMR-only artifacts:

- `*_series.csv`: assembled resonance series and higher-level fit exports
- `trace_diagnostics/*.png`: one diagnostic figure per fitted trace

VSM-only artifacts:

- `batch_hysteresis_overlay.png`: overlaid final hysteresis loops for the batch

## Modality Workflows

### ESR

Implemented flow:

- Parse Bruker `.dsc` descriptors plus sibling `.DTA` data.
- Apply explicit derivative baseline removal and optional Savitzky-Golay smoothing from the recipe.
- Integrate the processed derivative trace into absorption and cumulative area diagnostics.
- Detect up to two resonance windows.
- Fit either one full-trace derivative Lorentzian or two local peak-window Lorentzians.
- In `auto` mode, choose `single` or `split` by residual improvement.
- Export JSON, CSV, figure, reports, and ESR batch overlays/QC.

Core equations:

- Derivative fit model: `dL/dB = A * (-2x)/(1 + x^2)^2 + offset`
- Reduced field: `x = (B - B0)/gamma`
- Absorption model: `L(B) = A * gamma / (1 + x^2)`
- Peak-to-peak separation from the fitted linewidth parameter: `DeltaB_pp = 2 * gamma / sqrt(3)`
- Auto single vs split decision: `(SSR_single - SSR_split) / SSR_single`
- Local fit/data disagreement ratio: `abs(fit_local - local_data) / max(abs(fit_local), abs(local_data), 1e-12)`

Key extracted outputs:

- selected mode: `single` or `split`
- fitted center field `center_mT`
- fitted linewidth parameter `gamma_mT`
- R^2, residual summaries, convergence metadata, parameter diagnostics, and bound-hit flags
- detected peak windows and per-peak component fits
- total fit-derived area integral
- fit-local windowed integral
- local data-window integral
- diagnostic full-span integral
- fit validity fields such as fit scope, rejection reason, and selected-for-primary flags

ESR batch QC:

- normalized RMSE of the selected residual
- selected-fit R^2
- SNR from fitted amplitude versus residual standard deviation
- edge margin around fitted extrema
- uncertainty usability from center and gamma standard errors
- duplicate-run grouping by sample, replicate, nominal angle, and microwave frequency bucket
- best-run selection within duplicate groups based on edge margin, nRMSE, uncertainty score, SNR, and timestamp

### FMR

Implemented flow:

- Parse PhaseFMR `.log` files into one or more field-swept traces.
- Select the requested signal channel from the recipe.
- Apply explicit edge-baseline subtraction and optional smoothing.
- Fit each trace with either a single or double mixed derivative Lorentzian model.
- Perform component-level QC and accept or reject resonance components.
- Assemble accepted components into mode-specific resonance series.
- Fit Kittel and linewidth-vs-frequency models to each accepted series.
- Export per-trace, per-series, and per-physics artifacts.

Core equations:

- Single-trace mixed derivative Lorentzian with linear baseline:

```text
V(H) = component(H, Hres, DeltaH, S, A) + c0 + c1 * H
```

- With

```text
component(H) =
  S * [4 * DeltaH * (H - Hres)] / [4 * (H - Hres)^2 + DeltaH^2]^2
  - A * [DeltaH^2 - 4 * (H - Hres)^2] / [4 * (H - Hres)^2 + DeltaH^2]^2
```

- Double-resonance model: sum of two such components plus one shared linear baseline
- In-plane field-swept Kittel model: `f = gamma * sqrt(Hres * (Hres + Meff))`
- Linewidth model: `DeltaH = DeltaH0 + slope * f`
- Damping extraction: `alpha = slope * gamma_rad / (4 * pi)`
- Optional Gonzalez-Fuentes / Dumas / Garcia polarity averaging, when paired `+H` and `-H`
  sweeps are present for the same frequency and mode:
  `Hres_avg = (abs(Hres_pos) + abs(Hres_neg)) / 2`

Key extracted outputs:

- per-trace resonance field `H_res_mT`
- per-trace linewidth `DeltaH_mT`
- symmetric and antisymmetric amplitudes
- selected single or double fit mode
- accepted and rejected components with rejection reasons
- per-series resonance-field and linewidth trends versus frequency
- derived physics parameters: `gamma_GHz_per_T`, `g`, `M_eff_mT`, `alpha`, `DeltaH0_mT`
- polarity-pair diagnostics when enabled: `Hres_pos_mT`, `Hres_neg_mT`,
  `Hres_avg_mT`, `Hres_offset_mT`, and `Hres_split_mT`

Gonzalez-Fuentes polarity averaging is not a universal post-processing correction. It is
available only when the experiment includes paired positive-field and negative-field
FMR sweeps for the same frequency and mode. Single-polarity datasets continue with raw
`H_res_mT` Kittel fitting and the correction is marked skipped. Future FMR measurements
should be collected in both positive and negative field directions when high-confidence
g-factor extraction is desired.

FMR QC:

- residual RMSE fraction relative to signal magnitude
- amplitude SNR
- resonance field guard against sweep edges
- linewidth fraction limit relative to sweep span
- bound-hit checks on critical parameters
- shape-center consistency between fitted center and detected feature center

### VSM

Implemented flow:

- Parse VSM `.dat` loop exports into field, moment, uncertainty, and temperature arrays.
- Apply explicit preprocessing from the recipe.
- Split the loop into branch segments.
- Fit positive and negative high-field tails with linear models.
- Evaluate whether background slope subtraction should be accepted, rejected, or skipped.
- Optionally center the loop if requested by the recipe.
- Extract loop observables, trust metrics, and uncertainty estimates.
- Export JSON, trace CSV, summary CSV, figures, and VSM reports/overlays.

Core equations and definitions:

- Positive and negative tail fits: `m(H) = slope * H + intercept`
- Coercive fields from zero-crossing interpolation on the primary increasing and decreasing branches
- Remanence from interpolation of `m(H)` at `H = 0`
- Direct observables:
  - `Ms`: mean absolute saturation moment from selected tail windows
  - `Mr`: mean absolute remanence
  - `Hc`: mean absolute coercive field
  - `squareness = Mr / Ms`
  - `exchange_bias = loop_shift`
  - `vertical_shift`
  - loop area from branch overlap integration

Key extracted outputs:

- `Ms_emu`, `Mr_emu`, `Hc_mT`, `squareness`, `exchange_bias_mT`, `vertical_shift_emu`, `loop_area_emu_mT`
- branch-resolved observables such as `Hc-`, `Hc+`, `Mr+`, `Mr-`, `Ms+`, `Ms-`
- background fit slopes and intercepts for positive and negative tails
- centering offsets and whether centering was applied

VSM trust and QC fields:

- saturation confidence
- branch asymmetry
- switching complexity score and label
- ambiguity flags
- background mode: `slope_only`, `none`, or `rejected`
- background decision reason
- tail-fit R^2 values and soft/catastrophic threshold flags
- flatness gain and flatness-gain balance
- tail slope symmetry and saturation magnitude symmetry scores
- switching-width change after candidate correction
- zero-crossing candidate counts and coercive ambiguity counts

VSM uncertainty outputs:

- `ms_error`
- `mr_error`
- `hc_error`
- `hex_error`
- `squareness_error`
- `loop_area_error`

## Notes

- Recipes live under `recipes/esr/`, `recipes/fmr/`, and `recipes/vsm/`.
- The CLI and backend are currently the supported analysis path.
- Manifest and publication workflows are not exposed as runtime CLI features yet.
