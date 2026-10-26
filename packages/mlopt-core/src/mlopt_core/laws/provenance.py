"""Law provenance — which shipped rewrites are machine-admitted.

The library mixes two authorship classes and the yield measurement
(``project/retros/human-bar.md``) needs to tell them apart without
editing the laws themselves:

* **human-authored** — a law a person wrote into this package
  (including laws whose *discovery* was machine-assisted but which
  were hand-integrated before the admission machinery existed:
  ``select_mul``, ``softmax_fold``, ``glu_fold``, ``rms_norm_fold``);
* **machine-admitted** — an object the discovery side produced
  (``mlopt_discovery.object_synthesis`` constructions or
  census/proposal candidates), cleared the admission gauntlet
  (``mlopt_discovery.evidence.run_gauntlet``) and was then promoted
  verbatim into the library
  (``project/retros/promoted-laws.md``).

The marker is **registry-level**, not a per-law field: a ``Rewrite``
does not carry a provenance slot, so the classification lives here as
two name sets — :data:`MACHINE_SHIPPED` (the promoted laws now in
:data:`~mlopt_core.laws.ALL_RULES`) and :data:`MACHINE_STORED`
(gauntlet-cleared objects that stayed in the evidence store,
unshipped).  :func:`law_provenance` is *total* over rule names —
every shipped law resolves, and a name unknown to both sets classifies
as human-authored (the default authorship of the library).

The local evidence store (``mlopt_discovery.cache_dir()``) holds no
``lemmas`` rows — the
admission runs recorded their objects in ephemeral stores — so
:data:`MACHINE_STORED` is the honest ledger of *names* the committed
tests pin as ``usable`` (``tests/test_discovery_synthesis.py`` /
``test_discovery_gauntlet.py`` / ``test_discovery_autocond.py``)
plus the objects the promotion audit records as admitted
(``project/retros/promoted-laws.md``).
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "MACHINE_LAWS",
    "MACHINE_SHIPPED",
    "MACHINE_STORED",
    "PAY_MEASURED_SLOW",
    "law_provenance",
    "pay_status",
]

#: The machine-admitted objects promoted into the shipped bank —
#: the thirteen in ``ALL_RULES`` proper (``promoted-laws.md`` — the
#: promotion table's shipped
#: names, including the two objects that shipped as the strictly more
#: general view identities ``transpose_noop`` / ``chunk_single``, and
#: ``mul_unsq_nat`` — the census-minted two-sided unsqueeze naturality
#: whose guard was resolved by the adversarial sweep, see
#: ``project/retros/mul-unsq-nat.md``).  ``add_unsq_pad_r`` is the
#: first promotion whose paying site is a strictly-real (non
#: purpose-built) intake workload — ``RelativePositionBias`` — measured
#: through the auto-cond → gauntlet retry (``docs/results_realcorpus.md``).
#: ``conv_bn_fold`` is the first
#: of a second shipping class: a ``syn_*`` composite promoted *verbatim*
#: — its stored-closure records embed in
#: :mod:`mlopt_core.laws.minted` and its transported guard rebuilds at
#: import rather than being transcribed to ``cond``/``dspec``
#: (``project/retros/minted-composite-default.md``).
#:
#: The ``prim:synop_*`` family is the third: the ``zoo-prim`` mint's
#: 21 admitted folds, promoted verbatim out of the domain-side
#: registry into ``DEFAULT`` — torch-free pattern data whose op
#: denotations stay sink-side — plus the ``--prim-bootstrap`` mint's
#: 2 depth-2 ``prim:synop2_*`` folds composing over the shipped
#: ``synop_*`` vocabulary, plus the ``--prim-vocab`` run's 34
#: depth-3 ``prim:synop3_*`` folds composing over both shipped
#: tiers.  The 57 ``synop_*.expand`` / ``synop2_*.expand`` /
#: ``synop3_*.expand`` names are their shipped definitional
#: premises: machine-admitted axioms the folds carry as
#: ``rule.premises`` — shipped in the bank but NOT in
#: ``ALL_RULES`` (expanding the primitive would re-add the unfolded
#: spelling), so this set is not quite ``ALL_RULES``-bounded.
MACHINE_SHIPPED: frozenset[str] = frozenset(
    {
        "softsign_fold",
        "sdpa_fold_nomask",
        "sdpa_fold_div_nomask",
        "mul_unsq_nat",
        "mul_unsq_pad_l",
        "mul_unsq_pad_r",
        "sub_unsq_pad_l",
        "add_unsq_pad_r",
        "mul_reshape_inert_l",
        "transpose_noop",
        "chunk_single",
        "linear_channel_to_row_scale",
        "conv_bn_fold",
        # -- the zoo-prim mint: 21 folds + their 21 definitional
        # premises (synop_*.expand — shipped axioms, never saturation
        # rules)
        "prim:synop_03889d8fe1",
        "prim:synop_2b8cfd1c90",
        "prim:synop_39f76b66f8",
        "prim:synop_4c5296b3ba",
        "prim:synop_4c657cb2c0",
        "prim:synop_4f8fefdd38",
        "prim:synop_5147826430",
        "prim:synop_571fb768b0",
        "prim:synop_5e58f26875",
        "prim:synop_629f9ef13b",
        "prim:synop_62d36aefdb",
        "prim:synop_6c709f774a",
        "prim:synop_889d740672",
        "prim:synop_93e57c595e",
        "prim:synop_976f9836d7",
        "prim:synop_98a002d64b",
        "prim:synop_b246dcb838",
        "prim:synop_bb8c54c201",
        "prim:synop_d5b128282a",
        "prim:synop_f08a53f9a0",
        "prim:synop_f3290a1163",
        "synop_03889d8fe1.expand",
        "synop_2b8cfd1c90.expand",
        "synop_39f76b66f8.expand",
        "synop_4c5296b3ba.expand",
        "synop_4c657cb2c0.expand",
        "synop_4f8fefdd38.expand",
        "synop_5147826430.expand",
        "synop_571fb768b0.expand",
        "synop_5e58f26875.expand",
        "synop_629f9ef13b.expand",
        "synop_62d36aefdb.expand",
        "synop_6c709f774a.expand",
        "synop_889d740672.expand",
        "synop_93e57c595e.expand",
        "synop_976f9836d7.expand",
        "synop_98a002d64b.expand",
        "synop_b246dcb838.expand",
        "synop_bb8c54c201.expand",
        "synop_d5b128282a.expand",
        "synop_f08a53f9a0.expand",
        "synop_f3290a1163.expand",
        # -- the vocabulary-bootstrap mint: 2 depth-2 folds + their
        # definitional premises (synop2_*.expand unfolds to the FULL
        # manual denotation — the anchored derivation closes through
        # the shipped prim:synop_bb8c54c201)
        "prim:synop2_35bbd03301",
        "prim:synop2_d46808a970",
        "synop2_35bbd03301.expand",
        "synop2_d46808a970.expand",
        # -- the vocab-priced bootstrap mint: 34 depth-3 folds +
        # their definitional premises (synop3_*.expand unfolds to
        # the FULL manual denotation — the anchored derivation
        # closes through the shipped prim:synop*/prim:synop2_*
        # bank folds, up to three steps)
        "prim:synop3_04df5fd446",
        "prim:synop3_083e9c241f",
        "prim:synop3_0ca31bf1e5",
        "prim:synop3_0fa19d1ff8",
        "prim:synop3_1850a07696",
        "prim:synop3_207ef0ddb7",
        "prim:synop3_21b9729f32",
        "prim:synop3_24c5153e5e",
        "prim:synop3_2fcad5e803",
        "prim:synop3_304457cd8d",
        "prim:synop3_39dbcda988",
        "prim:synop3_426efb3fb1",
        "prim:synop3_42bb04032d",
        "prim:synop3_42f015e162",
        "prim:synop3_490109134e",
        "prim:synop3_4a015e46ce",
        "prim:synop3_4b31b4501a",
        "prim:synop3_4cdc06b88f",
        "prim:synop3_4cf8732640",
        "prim:synop3_50a12a0158",
        "prim:synop3_531bb1872b",
        "prim:synop3_543ad6d861",
        "prim:synop3_970223f613",
        "prim:synop3_9b8631df70",
        "prim:synop3_acc3c38681",
        "prim:synop3_c79a6fe687",
        "prim:synop3_d5452c1c56",
        "prim:synop3_d8a399f647",
        "prim:synop3_e8abc00db0",
        "prim:synop3_ed5918edde",
        "prim:synop3_efb0c0d7ce",
        "prim:synop3_f28f7ffd1a",
        "prim:synop3_f9c72e0a8f",
        "prim:synop3_fbbba382fc",
        "synop3_04df5fd446.expand",
        "synop3_083e9c241f.expand",
        "synop3_0ca31bf1e5.expand",
        "synop3_0fa19d1ff8.expand",
        "synop3_1850a07696.expand",
        "synop3_207ef0ddb7.expand",
        "synop3_21b9729f32.expand",
        "synop3_24c5153e5e.expand",
        "synop3_2fcad5e803.expand",
        "synop3_304457cd8d.expand",
        "synop3_39dbcda988.expand",
        "synop3_426efb3fb1.expand",
        "synop3_42bb04032d.expand",
        "synop3_42f015e162.expand",
        "synop3_490109134e.expand",
        "synop3_4a015e46ce.expand",
        "synop3_4b31b4501a.expand",
        "synop3_4cdc06b88f.expand",
        "synop3_4cf8732640.expand",
        "synop3_50a12a0158.expand",
        "synop3_531bb1872b.expand",
        "synop3_543ad6d861.expand",
        "synop3_970223f613.expand",
        "synop3_9b8631df70.expand",
        "synop3_acc3c38681.expand",
        "synop3_c79a6fe687.expand",
        "synop3_d5452c1c56.expand",
        "synop3_d8a399f647.expand",
        "synop3_e8abc00db0.expand",
        "synop3_ed5918edde.expand",
        "synop3_efb0c0d7ce.expand",
        "synop3_f28f7ffd1a.expand",
        "synop3_f9c72e0a8f.expand",
        "synop3_fbbba382fc.expand",
    }
)

#: Machine-constructed / machine-guarded objects that cleared the
#: admission gauntlet but were never written into the library — the
#: unshipped store population, by recorded object name.  Three groups:
#:
#: * guarded view-identity strips (auto-cond minted or oracle-named
#:   guards) — the ``*_id`` spellings the generic promoted laws
#:   subsume or leave unshipped;
#: * constructed folds — ``fold_object`` products (the metavar-attr
#:   ``sdpa_fold_div_nomask_g`` twin that stayed unshipped);
#: * constructed lifts/composites — ``lift_object`` /
#:   ``compose_objects`` products (carrier introductions and
#:   multi-block composites).
#:
#: Two name collisions bound what a *name-level* registry can say:
#:
#: * ``om_lift`` — the store holds a constructed ``om_lift`` object
#:   (``lift_object``, literal ``dim=-1`` pattern) while
#:   ``mlopt_carriers.om.OM_LIFT`` is a *human-authored* carrier law
#:   with a metavar-dim guard.  The name is excluded below: the
#:   registry resolves the shipped law's class, and the store
#:   object's provenance rides its ``ConstructedObject.construction``
#:   record, not this table.  Arm-level attribution must compare rule
#:   objects, not names, at exactly this seam.
#: * ``softsign_fold`` / ``sdpa_fold_nomask`` / ``sdpa_fold_div_nomask``
#:   — store objects under the same names as promoted laws; already
#:   classified machine via :data:`MACHINE_SHIPPED`, so omitted here.
MACHINE_STORED: frozenset[str] = frozenset(
    {
        # -- guarded view-identity strips (auto-cond / oracle-named)
        "mul_unsqueeze_l_id",
        "mul_unsqueeze_r_id",
        "sub_unsqueeze_l_id",
        "add_unsqueeze_l_id",
        "mul_transpose_l_id",
        "add_transpose_l_id",
        "mul_chunk_l_id",
        "mul_reshape_l_id",
        # -- constructed folds (fold_object)
        "sdpa_fold_div_nomask_g",
        # -- constructed lifts and composites (lift_object/compose)
        "aff_step_lift",
        "aff_scan2_lift",
        "affd_step_lift",
        "affd_scan2_lift",
        "affd_scan4_lift",
        "silu_fold_commuted",
        "channel_then_row_scale",
    }
)

#: Every machine-authored rewrite name — shipped promotions plus the
#: unshipped store population.
MACHINE_LAWS: frozenset[str] = MACHINE_SHIPPED | MACHINE_STORED


def law_provenance(rule: object) -> Literal["machine", "human"]:
    """Classify a rewrite's authorship: ``"machine"`` or ``"human"``.

    Accepts a rule name or a :class:`~mlopt_core.egraph.Rewrite`
    (reads ``.name``).  A name in :data:`MACHINE_LAWS` —
    machine-shipped promotions or machine-held store objects —
    reports ``"machine"``; anything else is ``"human"``, the
    library's default authorship.  The function is total over names
    by design: provenance is a property of the *registry*, not a flag
    each rule had to be edited to carry.
    """
    name = (
        rule if isinstance(rule, str) else getattr(rule, "name", rule)
    )
    return "machine" if name in MACHINE_LAWS else "human"


#: Shipped folds whose **pay** was measured on the delivery corpus and came
#: back slower — graphopt `docs/reports/pay_measured.md` (2026-10-25, 14
#: cells, same-executor rule).  Each loses on wall clock; the cited number
#: is the worst module it binds.  A fold here may be cited as *vocabulary*
#: (it is certified, machine-original) but never as a speed result.
#:
#: The rest are **not** paying either — they are *unmeasured*, which is not
#: evidence.  A fold whose denotation becomes a generated kernel leaves
#: this set: `prim_kernel_probe` measured 0.38x of the spec-walk for six of
#: the pointwise prims (2.6x faster).
PAY_MEASURED_SLOW: frozenset[str] = frozenset(
    {
        "prim:synop_4c657cb2c0",
        "prim:synop_5e58f26875",
        "prim:synop_bb8c54c201",
        "prim:synop2_35bbd03301",
        "prim:synop3_0ca31bf1e5",
        "prim:synop3_0fa19d1ff8",
        "prim:synop3_21b9729f32",
        "prim:synop3_24c5153e5e",
        "prim:synop3_4cdc06b88f",
    }
)


def pay_status(rule: object) -> str:
    """Return a shipped fold's measured-pay status.

    ``"measured-slow"`` — timed on the delivery corpus and slower than the
    same pipeline without it; ``"unmeasured"`` — a shipped prim with no
    wall-clock verdict (the cost model is not a measurement); ``"n/a"`` —
    not a shipped prim fold.

    Total over names, like :func:`law_provenance`: pay status is a property
    of the measurement ledger, not a flag a rule carries.
    """
    name = (
        rule if isinstance(rule, str) else getattr(rule, "name", rule)
    )
    # The ledger stores folds under their ``prim:`` name; the mint exposes
    # them bare (`SYNOP_SHIPPED`).  Accept either spelling.
    if name.startswith("synop") and not name.endswith(".expand"):
        name = "prim:" + name
    if name in PAY_MEASURED_SLOW:
        return "measured-slow"
    if name in MACHINE_SHIPPED and (
        name.startswith("prim:synop") or name.endswith(".expand")
    ):
        return "unmeasured"
    return "n/a"
