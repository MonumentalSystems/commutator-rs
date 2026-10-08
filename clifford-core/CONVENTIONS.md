# Clifford conventions

These conventions are part of the `clifford-core` compatibility contract.
Changing one requires a new golden-fixture schema and a migration plan.

## Signature and blades

- `Cl(p,q)` has `p` positive-squaring generators followed by `q`
  negative-squaring generators.
- Generator `i` is encoded by bit `i` of an unsigned blade bitmap.
- A blade is encoded by the bitwise union of its generators.
- Dense multivectors use ShortLex ordering: grade ascending, then bitmap
  ascending.
- The engine is non-degenerate. Zero-squaring generators and true projective
  `Cl(3,0,1)` are not currently supported.

`CliffordAlgebra::basis(i)` accepts a dense blade index, not a bitmap. Use
`bitmap_to_index[bitmap]` when translating a bitmap fixture or another
implementation.

## Products

The geometric product uses the canonical ascending-generator blade order.
The commutator and anticommutator include a factor of one half:

```text
[A,B] = (AB - BA) / 2
{A,B} = (AB + BA) / 2
```

Versor's current Cl(3,0), STA, and Cl(6,0) commutator functions return the raw
`AB - BA`. Consequently, a Versor commutator is twice the corresponding
`clifford-core` result after blade-layout conversion.

## Reversion and sandwich action

Reversion multiplies a grade-`k` blade by `(-1)^(k(k-1)/2)`. Rotor action is
the left sandwich product:

```text
X' = R X reverse(R)
```

For a simple normalized Euclidean bivector `B`, `rotor_from_bivector(B, theta)`
constructs `cos(theta/2) + B sin(theta/2)`. With this sign convention, an
`e12` rotor sends `e1` toward `-e2`. Versor's EGA rotor constructors use the
opposite exponential sign, so adapters must negate the angle or plane.

The closed form is not a general exponential for mixed-grade elements,
non-simple bivectors, or arbitrary signatures. The existing method remains a
compatibility API for Commutator. A zero bivector returns the identity rotor.

## Known layout adapters

- Versor EGA Cl(3,0) matches ShortLex exactly:
  `[0,1,2,4,3,5,6,7]`.
- Versor's legacy dense Cl(3,0) layout is
  `[0,1,2,4,3,6,5,7]`; the `e13` and `e23` slots are swapped.
- Versor Cl(6,0) full, bivector, and even layouts match the corresponding
  ShortLex subsets.
- Versor STA bivectors use bitmaps `[3,5,9,12,10,6]`, while the ShortLex
  order is `[3,5,6,9,10,12]`.

The version-one fixture records values verified against Versor commit
`93a5a1334ab77131e5ef6b88105cdaac57d267e6`. Consumers should vendor the
fixture rather than add a cross-repository path dependency.
