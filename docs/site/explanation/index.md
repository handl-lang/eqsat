# eqsat — explanations

<!-- provenance: docs: -->

The substrate side of the research stack — the pieces that live in
[`handl-lang/eqsat`](https://github.com/handl-lang/eqsat) after the
plan-0032 phase-4 split.

- [The e-graph and `dag_cost`](./egraph) — terms in, a saturated
  equivalence class out, and the cost models that rank its members.
- [Certificates](./certificates) — `Certificate`/`verify_certificate`:
  the replayable derivation that is the claim.
- [The ports](./ports) — `Source`/`Sink`/`Engine`/`Runner`: the seam
  between a substrate that knows no backend and the backends that know
  no search.

Cross-cutting concepts (the three costs, the game, the gauntlet) live
with [graphopt](/graphopt/).
