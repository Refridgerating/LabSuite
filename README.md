# LabSuite

LabSuite is a reproducible analysis platform for VSM, ESR, and FMR data. The
repository is organized as a layered scientific application: raw data ingestion,
explicit preprocessing, physics analysis and fitting, workflow orchestration,
and GUI presentation stay separate.

The installable Python package and current operator documentation live in
`labsuite/`.

## Repository Layout

- `labsuite/src/labsuite/core/`: shared schemas, preprocessing primitives,
  fitting abstractions, batch infrastructure, reporting, and exporters.
- `labsuite/src/labsuite/plugins/`: modality-specific ESR, FMR, and VSM
  parsers, preprocessing defaults, fitters, derived values, and services.
- `labsuite/src/labsuite/workflows/`: reusable single-file, batch, raw import,
  and measurement orchestration.
- `labsuite/src/labsuite/cli/`: the supported command-line operator surface.
- `labsuite/src/labsuite/gui/`: Qt presentation components. GUI code should
  call services and render results, not contain scientific formulas.
- `labsuite/metadata/`: sample registry and analysis ledgers.
- `labsuite/data/`: raw, intermediate, processed, and derived analysis data.
- `labsuite/recipes/`: YAML recipes for modality and sample-level analysis.

## Setup

From the repository root:

```powershell
cd labsuite
python -m pip install -e .
labsuite --version
```

For development tools and tests:

```powershell
python -m pip install -e ".[dev]"
pytest
```

## Operator Documentation

- `labsuite/README.md`: canonical CLI and workflow guide.
- `labsuite/metadata/README.md`: sample registry, measurement ledger, and
  processed ledger reference.
- `labsuite/data/README.md`: raw, intermediate, processed, and derived data
  folder conventions.
- `labsuite/recipes/README.md`: recipe workflow and default recipe roles.

## Architecture Rule

Every implemented analysis path follows:

```text
parse -> preprocess -> fit -> derive -> export
```

Parsing preserves raw imported data. Preprocessing and fitting are explicit,
recipe-driven steps. Derived values come from saved inputs and model outputs.
Important state must be serializable so CLI, GUI, and batch workflows can share
the same backend.
