# Workflows

## Purpose

This document defines the standard analysis workflow for all supported modalities in LabSuite.

It applies to:
- ESR
- FMR
- VSM

The scientific models will differ by modality, but the workflow contract must remain the same.

## Core workflow

All analysis must follow this sequence:

raw file
→ parse
→ raw structured dataset
→ preprocess
→ processed dataset
→ fit
→ fit result
→ derive parameters
→ export

No step should silently perform the work of another step.

## Workflow stages

### 1. Parse

Purpose:
- read the source file
- detect or validate format
- map raw columns and metadata into structured objects
- preserve source information

Parse must:
- return raw imported data
- preserve metadata where available
- avoid modifying the scientific meaning of the signal

Parse must not:
- smooth data
- subtract baselines
- normalize by thickness, area, mass, or magnetic volume
- fit peaks, loops, or resonance models
- silently discard points because they look bad

Output:
- raw structured dataset

### 2. Preprocess

Purpose:
- perform explicit signal cleanup and corrections before fitting

Typical examples:
- baseline subtraction
- smoothing
- resampling
- field or unit normalization
- high-field background correction
- alignment or trimming when explicitly requested

Preprocessing must:
- be explicit
- be recipe-driven
- be reproducible
- preserve provenance

Preprocessing must not:
- hide corrections inside parser logic
- perform final model fitting
- compute final derived scientific conclusions unless they are strictly preprocessing diagnostics

Output:
- processed dataset

### 3. Fit

Purpose:
- apply a chosen scientific model to the processed dataset

Examples:
- derivative Lorentzian fit for ESR
- absorption or derivative resonance fit for FMR
- hysteresis metric extraction or model fitting for VSM

Fitting must:
- use explicit model definitions
- use explicit initial guesses, bounds, and constraints where needed
- produce structured fit results

Fitting must not:
- silently re-preprocess data
- bury modality assumptions in shared core code
- store critical parameters only in GUI state

Output:
- fit result

### 4. Derive parameters

Purpose:
- convert fit results and metadata into physically meaningful quantities

Examples:
- g-factor
- linewidth
- effective damping-related trends
- coercivity
- remanence
- saturation-related quantities
- anisotropy-related parameters

Derived parameter calculation must:
- be traceable to fit outputs and metadata
- be deterministic
- remain separate from fitting logic where practical

Output:
- derived parameter result

### 5. Export

Purpose:
- save results and provenance in reusable formats

Exports may include:
- processed CSV
- fit result JSON
- aggregate CSV
- figures
- project/session state

Exports must include enough information to reproduce the analysis later.

## Universal rules

These rules apply to every workflow:
- raw data is preserved
- parsing is raw import only
- preprocessing is explicit
- fitting is separate
- derived parameters are separate
- workflows must run without the GUI
- all important state must be serializable

## Workflow types

### Single-file workflow

Use when:
- inspecting one dataset
- developing or testing a recipe
- validating a fit model

Sequence:
- load file
- parse
- preview raw data
- apply recipe
- fit
- inspect result
- export

### Folder workflow

Use when:
- processing many files with the same modality and recipe
- screening a directory of similar runs

Sequence:
- discover files
- assign modality and recipe
- run parse → preprocess → fit → derive → export for each file
- collect summary outputs

### Manifest workflow

Use when:
- different files need different recipes or metadata
- you need batch processing with explicit control

Manifest should define:
- file path
- modality
- sample ID
- recipe path
- grouping labels
- notes
- output target if needed

### Project workflow

Use when:
- multiple datasets belong to the same scientific study
- results need to be saved and revisited together

A project should preserve:
- linked source files
- metadata
- recipes used
- fit results
- derived results
- exported artifacts

## Modality-specific behavior

All modalities must follow the same workflow contract.

### ESR
Examples of modality-specific steps:
- derivative spectrum handling
- peak fitting
- linewidth extraction
- g-factor derivation

### FMR
Examples of modality-specific steps:
- resonance extraction across frequency or field
- Kittel fitting
- linewidth versus frequency workflows
- damping-related trend extraction

### VSM
Examples of modality-specific steps:
- loop background correction
- normalization choices
- coercivity and remanence extraction
- saturation-related analysis

These modality-specific operations belong in plugin packages, not in the shared workflow definition.

## Batch processing requirements

Batch processing must:
- continue when one file fails unless configured otherwise
- log failures clearly
- save per-file outputs
- save aggregate summaries
- preserve recipe and provenance information

Batch processing must not:
- stop because of one malformed file unless explicitly requested
- hide skipped files
- rely on GUI-only state

## Provenance requirements

Every completed workflow should save enough information to reconstruct the run:
- source file path or identifier
- modality
- metadata used
- preprocessing recipe
- fit recipe or model selection
- fit results
- derived parameters
- software version where practical
- timestamps where useful

## Design consequences

This document implies the following:
- GUI code is a thin layer over workflows and services
- CLI and GUI must share the same backend
- plugin code contains modality-specific science
- core code contains shared infrastructure only

## Anti-patterns

The following are not allowed:
- formulas inside GUI widgets
- smoothing or normalization hidden inside parsers
- modality-specific assumptions inside `core/`
- analysis state that exists only in memory
- silent preprocessing defaults that are not represented in recipes

## Summary

LabSuite workflows are built around one rule:

separate raw import, preprocessing, fitting, derived parameters, and export into distinct, explicit, reproducible stages.