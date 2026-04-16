# LabSuite

LabSuite is a Python desktop and CLI analysis platform for **VSM**, **ESR**, and **FMR** data.

The repository is intended to support:
- raw data import
- explicit preprocessing
- physics-based fitting and derived parameters
- batch processing
- reproducible exports
- a thin GUI over a shared backend

## Scope

This project is for **scientific data analysis**, not generic instrument control.

Primary goals:
- parse raw files into structured datasets
- keep preprocessing explicit and recipe-driven
- keep fitting separate from parsing and GUI code
- support both single-file and batch workflows
- make all important analysis state serializable

Non-goals for the early stages:
- instrument control
- cloud syncing
- database-backed multi-user workflows
- GUI-only logic that cannot be reproduced from CLI or saved recipes

## Approved technology stack

### Core language
- Python 3.11+

### GUI
- **PySide6** for the desktop application and Qt widgets
- **pyqtgraph** for fast interactive plotting inside the GUI

### Scientific computing
- **NumPy** for arrays and numerical operations
- **SciPy** for signal processing, interpolation, optimization, and numerical utilities
- **lmfit** for higher-level nonlinear fitting, constrained parameters, and fit reporting

### Data and configuration
- **Pydantic** for validated schemas and structured models
- **PyYAML** for recipe and config files
- **pandas** for tabular data handling and aggregate exports

### CLI and developer tooling
- **Typer** for the command-line interface
- **Rich** for readable CLI output
- **pytest** for tests
- **ruff** for linting
- **mypy** for optional type checking
- **PyInstaller** for packaging desktop builds

## Why these packages

- **PySide6** is the official Qt for Python binding, and Qt Widgets provides the classic desktop UI elements needed for tables, dialogs, panes, and tool windows. :contentReference[oaicite:0]{index=0}
- **pyqtgraph** is the primary plotting layer because `PlotDataItem` is designed for interactive 2D plotting and supports display-oriented operations such as transformation and decimation, which are useful for dense lab traces. :contentReference[oaicite:1]{index=1}
- **lmfit** is used over raw SciPy fitting alone because it provides a higher-level interface for nonlinear fitting, parameter handling, constraints, and model-based workflows built on `scipy.optimize`. :contentReference[oaicite:2]{index=2}
- **Typer** is used for the CLI because it is built around Python type hints and is well-suited to structured command trees for batch and export workflows. :contentReference[oaicite:3]{index=3}

## Workflow

All analysis workflows should follow the same high-level pipeline:

raw file
→ parse
→ raw structured dataset
→ preprocess
→ processed dataset
→ fit
→ derived parameters
→ export

This workflow applies to ESR, FMR, and VSM.

### Rules
- parsing reads and structures raw data only
- preprocessing performs explicit cleanup and corrections
- fitting applies the selected model
- derived parameters are calculated after fitting
- export writes results, metadata, and provenance to disk

Each stage must remain separate and testable.

## Architectural rules

Read `AGENTS.md` first.

Hard requirements:
- do **not** bury formulas inside GUI widgets
- do **not** mix parser cleanup with physics correction
- do **not** hardcode instrument-specific assumptions into `core/`
- do **not** make analysis state live only in memory

Interpretation:
- parsers read raw files only
- preprocessing is explicit and recipe-driven
- fitting is separate from preprocessing
- derived parameters are separate from fitting
- GUI calls services, it does not implement the science
- all meaningful analysis state must be saved or serializable

## Repository structure

```text
src/labsuite/
├─ app/          # application entrypoints and startup
├─ gui/          # Qt widgets, windows, models, plotting views
├─ cli/          # Typer commands
├─ core/         # shared schemas, preprocessing, fitting abstractions, export, batch
├─ io/           # format sniffing and shared parsers
├─ plugins/      # modality-specific science: ESR, FMR, VSM
└─ workflows/    # single-file, batch, manifest, publication workflows
