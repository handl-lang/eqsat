"""masked_fill sdpa-fold synthesis — numeric-leg abstention.

Regression for the graphopt
``test_masked_fill_sdpa_fold_emits_only_with_seed`` failure: the torch
concrete-eval env (``mlopt_torch.meta_eval.TorchConcreteEval.make_env``)
mints an fp64 tensor for EVERY leaf — including the boolean mask
``masked_fill`` requires — so evaluating the bound instance raises
``RuntimeError: masked_fill_ only supports boolean masks``.  Every
instantiation vetoed and ``synthesize_rules`` emitted zero candidates
even though the composition is valid.

The numeric leg now vetoes only the unevaluable *instance*; when no
tried instance yields a numeric verdict it abstains to structural-only
(the same degradation as an unbound op) instead of rejecting a
candidate the backend cannot measure.  This file reproduces the
synthesis call torch-free: :class:`_BoolMaskEval` mints floats for
every leaf and raises on a non-bool ``masked_fill`` mask, exactly the
env deficiency torch exhibits.
"""

from mlopt_core import laws as R
from mlopt_core import meta
from mlopt_core.ir import Const, Op, Param, TensorType, Var, op_repr


def _T(*shape):
    return TensorType(tuple(shape))


def _sdpa_fold(name):
    return next(r for r in R.SDPA_FOLD_RULES if r.name == name)


def _masked_fill_attention_seed():
    """matmul(softmax(masked_fill(mul(0.5, q@kᵀ), mk, -inf), -1), v)."""
    q = Var("q", _T(5, 4))
    k = Var("k", _T(7, 4))
    v = Var("v", _T(7, 6))
    mk = Var("mk", _T(5, 7))
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    term = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "masked_fill",
                Op.make("mul", Const(0.5), scores),
                mk,
                Const(float("-inf")),
            ),
            dim=-1,
        ),
        v,
    )
    return term


class _BoolMaskEval:
    """``ConcreteEval`` stub reproducing the torch env deficiency.

    ``make_env`` hands every typed leaf a float — the fp64-for-all
    contract ``TorchConcreteEval`` has — so ``masked_fill``'s mask
    operand is never a bool and its eval raises ``RuntimeError`` on
    every instance.  Only the ops on the eval path down to
    ``masked_fill`` need evaluators (``matmul``/``transpose``/``mul``);
    ``softmax``/``sdpa``/``logical_not`` are never reached because the
    mask raises first — and the claimed output is never evaluated
    either, since no instance produces a numeric verdict.
    """

    def make_env(self, leaf_shapes):
        return {
            leaf: float(i + 1)
            for i, leaf in enumerate(sorted(leaf_shapes, key=repr))
        }

    def eval_term(self, term, env):
        if isinstance(term, Const):
            return term.value
        if isinstance(term, (Var, Param)):
            return env[term]
        if isinstance(term, Op):
            args = [self.eval_term(a, env) for a in term.args]
            if term.op == "masked_fill":
                x, m, _v = args
                if not isinstance(m, bool):
                    raise RuntimeError(
                        "masked_fill only supports boolean masks"
                    )
                return x
            if term.op == "matmul":
                return 0.0
            if term.op == "transpose":
                return args[0]
            if term.op == "mul":
                return args[0] * args[1]
            raise KeyError(term.op)
        raise TypeError(term)

    def allclose(self, a, b, tol):
        return a == b


def _provenance_in(derived, parents):
    return [d for d in derived if meta.provenance(d) == parents]


def test_masked_fill_sdpa_fold_emits_despite_bool_mask_env(monkeypatch):
    """comm_mul ∘ sdpa_fold_masked_fillmul emits with a seed even when
    the concrete-eval env cannot type the boolean mask leaf.

    Without seeds the guarded pair is unreachable (the Const -inf fill
    defeats the bounded instantiation pool).  With the seed the
    seed-guided path mints the composite on the fired witness; the
    numeric leg cannot evaluate ANY instance — every one raises inside
    ``masked_fill`` on the fp64-minted mask — so it abstains to
    structural-only and the rule emits."""
    seed = _masked_fill_attention_seed()
    rules = [R.COMM_MUL, _sdpa_fold("sdpa_fold_masked_fillmul")]
    parents = ("comm_mul", "sdpa_fold_masked_fillmul")

    monkeypatch.setattr(meta, "_concrete_eval", _BoolMaskEval())

    bare = meta.synthesize_rules(rules, fuel=2000)
    assert not _provenance_in(bare, parents)

    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    hits = _provenance_in(derived, parents)
    assert hits, "masked_fill sdpa fold not synthesized"
    d = hits[0]
    assert d.check is not None and d.derive is not None
    assert "logical_not" in op_repr(d.rhs)

    applied = meta.apply_rewrite_at(d, seed, ())
    assert applied is not None and applied.op == "sdpa"
    assert applied.attrs["scale"] == 0.5
    # fill-mask became the kernel's keep-mask via logical_not
    assert applied.args[3].op == "logical_not"
