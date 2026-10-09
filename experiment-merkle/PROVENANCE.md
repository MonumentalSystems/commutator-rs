# Provenance

Monumental Systems' private repositories contain an earlier SHA-256
field-result tree. Its history includes the HarmonicRust skeleton at
`d16d7fe3efe350fb9417533eba16116877ad84f5`, implementation at
`6404da21e1208d797c90db87ca6fc203444ac7a2`, extraction at
`ceb0a6309550284f386d02269f06f4a0668710c5`, and its later Commutator import
at `4aa843af99caed0f4c7286d43692efef342a2182`.

`experiment-merkle` is a redesigned public v1 informed by operational use
of that system, not a wire-compatible copy. The legacy format allows proof
orientation and the proof-carried leaf hash to determine verification without
cryptographically deriving both from the challenged index and bytes. It also
lacks context, leaf-count, and domain binding.

The public v1 contract deliberately changes the root and proof formats to
bind those values, use fallible checked APIs, and impose allocation limits.
Existing Commutator deployments remain untouched and must negotiate a future
v1 adapter explicitly rather than silently interpreting legacy roots as v1.

This implementation is released under the workspace MIT license by its
contributors.
