# clifford-layers

Framework-neutral neural-network layers over [`clifford-core`](../clifford-core).

The crate contains:

- geometric-product linear layers;
- one- and two-dimensional Clifford convolutions;
- blade interleaving helpers and vector-valued SiLU activations;
- Clifford group and batch normalization;
- optional one-dimensional Clifford Fourier convolution via the `fourier`
  feature.

It deliberately contains no model runtime, optimizer, autodiff framework,
distributed-computing protocol, GPU backend, or experiment lifecycle.
Inputs and parameters are flat `f32` buffers with shapes documented on each
operation. Shape mismatches panic so programming errors fail immediately.

```rust
use clifford_core::CliffordAlgebra;
use clifford_layers::CliffordLinear;

let algebra = CliffordAlgebra::cl3();
let mut layer = CliffordLinear::new(algebra, 1, 1, false);
layer.weights.fill(0.0);
layer.weights[0] = 1.0;

let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
assert_eq!(layer.forward(&input), input);
```

## Fourier feature

```bash
cargo test --features fourier
```

## Provenance

The implementation was extracted from Monumental Systems' HarmonicRust
`harmonic-core` crate at source commit
`c20fd04956f987f9a00d53c78728d8069f0a1589`. Those modules were pure-Rust
ports inspired by the public `cliffordlayers` design. This crate uses
`clifford-core`'s ShortLex blade ordering and geometric-product tables as its
sole algebra convention. The extraction corrected the dense block-kernel
orientation by testing it directly against `CliffordAlgebra::geometric_product`.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the upstream Microsoft
MIT notice and suggested research citations.

## License

MIT
