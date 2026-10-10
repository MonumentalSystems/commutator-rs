# quantum-shadows

`quantum-shadows` is a small, dependency-free reference implementation of
local random-Pauli classical-shadow estimators. It consumes measurement records
from simulators or hardware and estimates many Pauli observables without
constructing an exponentially large density matrix.

```rust
use quantum_shadows::{MeasurementBasis, Pauli, PauliString, ShadowDataset, Snapshot};

let records = vec![
    Snapshot::try_new(vec![MeasurementBasis::Z], vec![1])?,
    Snapshot::try_new(vec![MeasurementBasis::Z], vec![1])?,
];
let shadows = ShadowDataset::try_new(1, records)?;
let z = PauliString::try_new(vec![Pauli::Z])?;
assert_eq!(shadows.mean(&z)?, 3.0); // a valid finite-sample shadow estimate
# Ok::<(), quantum_shadows::ShadowError>(())
```

The factor of three is the inverse local-Pauli measurement channel. Unbiasedness
is over uniformly random X/Y/Z basis choices; a deliberately basis-selected
sample, as above, need not itself lie in the physical expectation interval.
Random-basis generation stays outside the crate so experiments can control and
record their RNG exactly.

The crate provides mean, standard-error, median-of-means, linear-observable,
and conservative Hoeffding-radius calculations. It does not claim full quantum
process tomography, gate-set tomography, or detector-error mitigation.

## License

MIT.

