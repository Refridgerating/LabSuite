# AGENTS.md

## Purpose

This repository is for a lab analysis platform for VSM, ESR, and FMR data.

The codebase must be developed as a layered scientific application with clear separation between:
- data ingestion
- preprocessing
- physics analysis and fitting
- workflow orchestration
- GUI presentation

The goal is reproducible, testable, batch-capable scientific software. Not a monolithic desktop script.

---

## Non-negotiable architectural rules

### 1. Do not bury formulas inside GUI widgets
Physics equations, fitting logic, derived parameter calculations, and scientific transforms must never live inside Qt widgets, dialogs, plot classes, button handlers, or view models.

Allowed:
- GUI triggers service calls
- GUI renders results
- GUI edits recipes and parameters

Not allowed:
- Kittel fitting inside a button callback
- linewidth extraction inside a plot widget
- VSM normalization inside table model code
- ESR g-factor calculation inside dialog logic

All scientific logic belongs in:
- `src/labsuite/core/`
- `src/labsuite/plugins/<modality>/`
- `src/labsuite/workflows/`
- later `src/labsuite/studies/`

---

### 2. Do not mix parser cleanup with physics correction
Parsing must only:
- read files
- detect format
- map columns
- preserve metadata
- return raw structured data

Parsing must not:
- subtract baselines
- smooth data
- normalize by thickness, area, mass, or magnetic volume
- convert to publication-ready corrected traces
- silently drop points because they look noisy
- apply instrument-specific “fixes” unless explicitly labeled and logged as import transforms

Scientific cleanup belongs in explicit preprocessing steps, driven by recipes and saved in provenance.

Rule:
- parser output = raw imported dataset
- preprocessing output = corrected dataset
- fitting output = model result
- derived output = physical parameters

These stages must remain separate.

---

### 3. Do not hardcode instrument-specific assumptions into `core/`
The `core/` package must remain modality-agnostic and instrument-agnostic.

`core/` may contain:
- shared data models
- recipe schemas
- preprocessing utilities
- fitting abstractions
- batch infrastructure
- exporters
- generic math utilities

`core/` must not contain:
- assumptions about ESR derivative lineshapes
- VSM high-field slope conventions tied to one machine
- FMR field units tied to one vendor
- file header assumptions from a specific instrument
- sample naming rules specific to one lab

Instrument and modality behavior belongs in:
- `src/labsuite/plugins/esr/`
- `src/labsuite/plugins/fmr/`
- `src/labsuite/plugins/vsm/`

If logic depends on modality, vendor, export format, or instrument quirks, it does not belong in `core/`.

---

### 4. Do not make analysis state live only in memory
All meaningful analysis state must be serializable, reproducible, and recoverable from disk.

Must be saved explicitly:
- raw file references
- imported metadata
- preprocessing recipe
- fit recipe
- fitted parameters
- bounds and constraints
- derived parameters
- export settings
- software version where feasible
- timestamps where useful

Not allowed:
- analysis that only exists in widget state
- fit settings stored only in temporary GUI controls
- unsaved batch queue assumptions
- manual tweaks that cannot be reproduced later

Every important transformation should be representable as data.

Preferred formats:
- YAML for recipes
- JSON for structured results
- CSV for tabular exports
- project/session files for grouped workflows

---

## Required layering

### GUI layer
Responsible for:
- user interaction
- dataset browsing
- plot display
- recipe editing
- fit preview presentation
- batch queue visualization

GUI must call services. GUI must not perform domain science directly.

### Core layer
Responsible for:
- shared schemas
- reusable preprocessing primitives
- generic fitting framework
- batch orchestration
- exports
- logging
- project/session structures

### Plugin layer
Responsible for:
- modality-specific parsing
- modality-specific preprocessing defaults
- fit models
- derived parameter formulas
- instrument-specific import behavior
- science-specific services

### Workflow layer
Responsible for:
- single-file workflows
- folder workflows
- manifest workflows
- publication export workflows
- later angle, temperature, and multimodal study workflows

---

## Scientific reproducibility requirements

When implementing any analysis step:
1. preserve the raw input
2. log or serialize the transform applied
3. keep preprocessing explicit
4. keep fitting explicit
5. make derived values traceable to inputs and model outputs
6. support batch execution without GUI interaction

If a result cannot be reproduced from saved configuration and source data, the design is wrong.

---

## Rules for Codex and contributors

When adding code:
- prefer small service functions over large classes unless state is meaningful
- keep modality-specific logic inside plugin packages
- write code so CLI and GUI can share the same backend
- add tests for parsers, preprocessing, fits, and derived parameters
- document assumptions explicitly
- avoid silent corrections
- Use ruff to check work

When uncertain about placement:
- ask whether the code is presentation, shared infrastructure, or modality-specific science
- place it in the narrowest layer that is still reusable

---

## Red flags

Stop and refactor if you see:
- formulas in Qt widget files
- parser functions performing smoothing or normalization
- `core/` importing modality-specific fitters
- state stored only in GUI objects
- hidden preprocessing defaults not represented in recipes
- one-off vendor logic leaking into shared modules

---

## Preferred design principle

Thin GUI, explicit workflows, reproducible science, plugin-based modality separation.