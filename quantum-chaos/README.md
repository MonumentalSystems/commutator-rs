# quantum-chaos

`quantum-chaos` provides checked, deterministic diagnostics over supplied
energy levels. It is not an eigensolver and it does not infer a dynamical
phase from one statistic. The crate keeps raw spectra, unfolding choices, and
finite-sample reference comparisons explicit.

Included diagnostics:

- adjacent-gap ratios and their sample mean;
- absolute comparison with the Poisson and GOE-surmise reference means;
- global-affine and explicitly windowed local-gap unfolding;
- normalized spectral form factor `K(t) = |sum_n exp(-i E_n t)|^2 / N`;
- number-count variance over caller-supplied interior windows.

```rust
use quantum_chaos::{Spectrum, UnfoldingPolicy};

let raw = Spectrum::try_from_sorted(vec![1.0, 2.1, 3.0, 4.2])?;
let ratios = raw.adjacent_gap_ratios(1e-12)?;
assert_eq!(ratios.ratios().len(), 2);

let unfolded = raw.unfold(UnfoldingPolicy::Affine { minimum_gap: 1e-12 })?;
let form_factor = unfolded.spectral_form_factor(&[0.0, 1.0])?;
assert_eq!(form_factor[0], 4.0);
# Ok::<(), quantum_chaos::ChaosError>(())
```

## Interpretation

For independent exponential spacings, the mean adjacent-gap ratio is
`2 ln(2) - 1`. The common GOE ratio-distribution surmise gives
`4 - 2 sqrt(3)`. The crate reports distances to those means; closeness alone
is not a hypothesis test, and symmetry sectors must be separated before
level statistics are interpreted.

The affine policy only fixes the global mean spacing. The local-gap policy
normalizes each gap by a moving average of neighboring gaps, then integrates
the normalized gaps. Small windows can suppress the very long-wavelength
fluctuations one intends to measure, so the selected half-window is retained
in the returned metadata.

Number variance is defined here as the population variance of counts in
half-open intervals `[origin, origin + L)`. Callers provide origins and every
window must fit inside the supplied unfolded sequence, making the finite-size
edge policy explicit. Spectral rigidity is intentionally omitted until a
similarly unambiguous finite-sample estimator is selected.

## License

MIT.
