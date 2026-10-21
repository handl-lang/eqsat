# Certificates — the replayable derivation

<!-- provenance: eqsat:packages/mlopt-core/src/mlopt_core/egraph/{certs,proof}.py@e0633d5 -->

A certificate is the artifact a search produces that outlives the
search: a replayable derivation from the source term to the extracted
term. The certificate, not the timing, is the claim.

## The model

- `Certificate` — the derivation: an ordered list of `CertStep`s (a rule
  name, the e-class it fired on, the resulting merge) over `ProofEdge`s.
- `EGraph.certificate(src, dst, root_eid=...)` — extract the derivation
  the saturated graph witnesses between two terms.
- `verify_certificate(term, cert, *, strict=False)` — replay the
  certificate *standalone*: apply each step to the source term and check
  the claimed result lands in the same class. A certificate that does
  not replay raises `CertificateVerificationError` — no trust in the
  engine that emitted it.
- `cert_to_data` / `cert_from_data` — the transport form, so a
  certificate can cross a wire, a cache, or a backend boundary.

## Two places certificates do the work

1. **Delivery.** Every program the pipeline lowers ships its
   certificate — `search → extract → certify` — so downstream consumers
   verify the derivation, not the optimizer's reputation. See
   [your first `search()`](/graphopt/tutorials/first-search).
2. **Admission.** The derivability check in the
   [gauntlet](/graphopt/explanation/the-gauntlet) is a certificate
   replay: `verify_law(lhs, rhs, rules)` (the domain side, in
   `graphopt`'s `mlopt_discovery`) interns both sides in a fresh small
   e-graph, saturates under the known laws, and replays the witness *in
   both directions* — a merge is a genuine derivation, never a trusted
   stub. This substrate supplies the e-graph and the replay; the
   admission policy is the domain's.

## The honest boundary

A certificate proves *equality of the derivation*, nothing else: it says
the extracted program computes the same thing as the source. Whether the
extraction was worth it is a [measured](/graphopt/explanation/three-costs)
question the certificate deliberately does not answer.
