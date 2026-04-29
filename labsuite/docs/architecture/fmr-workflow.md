# FMR Workflow Notes

The FMR workflow is branch-oriented. A field-swept PhaseFMR/NanOsc trace is first fit
spectrally, accepted resonance components are assigned to frequency-continuous branches, and
branch-level physics is fit only after that separation.

## Geometry and Field Polarity

Geometry and field polarity are separate metadata fields.

- `geometry` describes field orientation relative to the film plane: `IP` or `OOP`.
- `field_polarity` describes applied field sign or direction: `positive`, `negative`, or `unknown`.

Do not use IP/OOP as a substitute for positive/negative sweep direction. Positive and negative
sweeps are commonly produced as separate raw files, so polarity matching is metadata-driven after
per-file processing.

## Multi-Peak Spectral Fits

PhaseFMR/NanOsc signals are field-modulated and derivative-like. The spectral model is a sum of
one to three mixed absorptive/dispersive derivative Lorentzian components plus a shared background.
The background can be linear or quadratic.

Auto mode fits one peak first, inspects structured residuals, then tries additional components when
residual shoulders justify them. Extra peaks are rejected or marked low confidence when linewidth,
amplitude, separation, uncertainty, or residual improvement is not credible.

## Branch Tracking

Branch tracking assigns accepted per-trace peaks to stable `branch_id`s using continuity in
`Hres(f)` with linewidth and jump penalties. Ambiguous or weak peaks may remain low confidence or
unassigned instead of being forced into a branch.

Each branch produces its own `Hres(f)` and `deltaH(f)` series.

## Kittel and Linewidth Fits

Branch-level Kittel models report FMR-derived field-like magnetization as `mu0_Meff_T`. For
Quantum Design/PhaseFMR comparison, the same value may also be exported as
`mu0_Ms_apparent_T`, but it is not automatically a true saturation magnetization.

VSM saturation magnetization must be reported separately as `Ms_A_per_m` and `mu0_Ms_T`.
Comparing VSM `mu0_Ms_T` and FMR `mu0_Meff_T` is a diagnostic, not an identity.

Locked branch-specific `g` or `gamma/2pi` values are preferred for trusted reporting when the
frequency span is limited. Floating-`g` fits are diagnostic unless explicitly promoted and should be
treated cautiously, especially if `g`, `Meff`, and `Hk` are all free.

Linewidth fits use:

```text
mu0_deltaH(f) = mu0_deltaH0 + slope * f
```

`alpha_eff` is computed from the branch gamma. Fewer than four high-confidence linewidth points is
reported as diagnostic or low confidence.

## Positive/Negative Matching

Positive/negative matched datasets are built after branch fitting by matching processed branch
points with:

- same sample
- same geometry
- same replicate when available
- same branch
- frequency within tolerance
- opposite field polarity

Matched averages use absolute resonance fields:

```text
Hres_avg = (abs(Hres_positive) + abs(Hres_negative)) / 2
Hres_asymmetry = abs(abs(Hres_positive) - abs(Hres_negative))
```

Both original values, `matched_pair_id`, `matching_confidence`, and asymmetry are preserved.
