# Third-party notices

## Microsoft `cliffordlayers`

Several layer designs in this crate derive from or were inspired by
[Microsoft `cliffordlayers`](https://github.com/microsoft/cliffordlayers),
including Clifford block kernels, rotation kernels, vector activations,
blade-aware normalization, and spectral Clifford convolution. HarmonicRust
ported those concepts to Rust; this crate was then extracted from HarmonicRust
commit `c20fd04956f987f9a00d53c78728d8069f0a1589` and adapted to
`clifford-core` conventions.

The upstream project is distributed under the following license:

> MIT License
>
> Copyright (c) Microsoft Corporation.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Suggested research citations

- Johannes Brandstetter, Rianne van den Berg, Max Welling, and Jayesh K. Gupta,
  “Clifford Neural Layers for PDE Modeling,” arXiv:2209.04934, 2022.
- David Ruhe, Jayesh K. Gupta, Steven de Keninck, Max Welling, and Johannes
  Brandstetter, “Geometric Clifford Algebra Networks,” arXiv:2302.06594, 2023.
