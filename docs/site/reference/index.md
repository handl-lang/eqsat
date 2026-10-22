# eqsat — reference

<!-- provenance: docs: -->

The substrate contract surface, pinned against the phase-1 frozen
manifest (`graphopt:docs/public_api.md` — `API_VERSION = "0.1.0"`).

- [The e-graph](./egraph) — the `EGraph` protocol signature-level:
  `add_term`/`find`/`rebuild`/`run`/`extract_best`, the `Rewrite`
  record, and the certificate codec.
- [The cost algebra](./cost) — `dag_cost` and the shipped cost-model
  families, signature by signature.
- [The ports](./ports) — `Source`/`Capabilities`/`Sink`/`Engine` +
  the rest of the port vocabulary, and `signature_conforms`.
- [The frozen API slice](./frozen-api) — the substrate rows of the
  manifest, vendored verbatim.

The conceptual accounts (why these shapes, not just which) live under
[explanation](../explanation/); the domain-side rows of the manifest
are [graphopt's](/graphopt/reference/frozen-api) and the game rows are
[gamekit's](/gamekit/reference/frozen-api).
