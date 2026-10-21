"""Substrate-pure tests ported from ``graphopt:tests/test_compositional_cache.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

from mlopt_core.egraph import EGraph

from mlopt_core.ir import IR, Const, Op, Param, TensorType, Var

from mlopt_core.pipeline import SearchResult

from mlopt_orchestrator import Compositional, Optimizer, structural_key

from mlopt_orchestrator.optimize import (
    _block_key,
    _cache_entry,
    _cache_replay,
    _derive_dedup,
    _leaf_equal,
    _param_order,
    _param_shapes_ok,
    _remap_term,
    _term_param_names,
)


def _t(*shape):
    return TensorType(tuple(shape))

def _fake_result(term, ir, stats=None):
    return SearchResult(
        ir=ir,
        eg=EGraph(),
        root_eid=0,
        term=term,
        param_values={},
        stats=stats or {},
    )

def _mk_ir(term, inputs, params):
    return IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params={p.name: p for p in params},
    )

def test_structural_key_renames_leaves():
    """Same structure, different leaf names → identical key."""
    x = Var("x", _t(4, 8))
    t1 = Op.make("matmul", x, Param("p_a", _t(8, 8)))
    t2 = Op.make(
        "matmul", Var("input0", _t(4, 8)), Param("w1", _t(8, 8))
    )
    assert structural_key(t1, inputs=[x]) == structural_key(
        t2, inputs=[Var("input0", _t(4, 8))]
    )

def test_structural_key_distinguishes_structure():
    x = Var("x", _t(4, 8))
    y = Var("y", _t(4, 8))
    p = Param("p", _t(8, 8))

    # op name
    assert structural_key(Op.make("matmul", x, p)) != structural_key(
        Op.make("mul", x, p)
    )
    # Var position (non-commutative operand order)
    assert structural_key(
        Op.make("sub", x, y), inputs=[x, y]
    ) != structural_key(Op.make("sub", y, x), inputs=[x, y])
    # Const value AND const type (``Op.make`` normalises positional
    # scalars, so the type tag is exercised on bare Const leaves)
    assert structural_key(Op.make("pow", x, Const(2))) != structural_key(
        Op.make("pow", x, Const(3))
    )
    assert structural_key(Const(2)) != structural_key(Const(2.0))
    # Var arity — an extra (even unused) input is a different block
    assert structural_key(
        Op.make("matmul", x, p), inputs=[x]
    ) != structural_key(Op.make("matmul", x, p), inputs=[x, y])
    # Param shape
    assert structural_key(
        Op.make("matmul", x, Param("p", _t(8, 8)))
    ) != structural_key(Op.make("matmul", x, Param("p", _t(8, 4))))
    # dtype rides in via leaf_meta (TensorType is shape-only)
    meta_f32 = {"p": "torch.float32", "x": "torch.float32"}
    meta_f64 = {"p": "torch.float64", "x": "torch.float32"}
    t = Op.make("matmul", x, p)
    assert structural_key(t, inputs=[x], leaf_meta=meta_f32) != (
        structural_key(t, inputs=[x], leaf_meta=meta_f64)
    )

def test_structural_key_param_positions_canonical():
    """matmul(p_a, p_b) and matmul(p_b, p_a) share a key — the slot
    correspondence is positional, which is exactly what the remap uses."""
    pa, pb = Param("a", _t(4)), Param("b", _t(4))
    assert structural_key(Op.make("matmul", pa, pb)) == structural_key(
        Op.make("matmul", pb, pa)
    )
    # and first-occurrence order records the distinct slots
    assert _param_order(Op.make("matmul", pa, pb)) == ("a", "b")

def test_structural_key_edge_leaves():
    x = Var("x", _t(4))
    p = Param("p", _t(4))
    # shared subterm — memo-hit branch (terms are interned)
    shared = Op.make("neg", x)
    t = Op.make("add", shared, shared)
    assert isinstance(structural_key(t), tuple)
    # Var absent from inputs falls back to its name
    k_named = structural_key(Op.make("mul", x, p))
    k_other = structural_key(Op.make("mul", Var("z", _t(4)), p))
    assert k_named != k_other
    # sequence / mapping / non-term roots
    assert structural_key([x, p]) != structural_key([p, x])
    assert structural_key({"w": p}) == structural_key({"w": p})
    assert structural_key(42) == structural_key(42)
    # unhashable leaf — memo lookup fails closed to a repr key
    assert structural_key({1, 2}) == structural_key({1, 2})
    # attrs canonicalise: term-valued, list-valued and unhashable attrs
    q = Param("q", _t(4))
    a = Op.make(
        "fake",
        x,
        sub=Op.make("neg", p),
        dims=[1, 2],
        blob={1, 2},
        table={"k": p},
        n=3,
        validate=False,
    )
    b = Op.make(
        "fake",
        x,
        sub=Op.make("neg", q),
        dims=[1, 2],
        blob={1, 2},
        table={"k": q},
        n=3,
        validate=False,
    )
    assert structural_key(a, inputs=[x]) == structural_key(b, inputs=[x])

def test_remap_term_renames_params_and_vars():
    x, w = Var("x", _t(4)), Var("w", _t(4))
    term = Op.make("matmul", x, Op.make("mul", Param("a", _t(4)), Const(2)))
    out = _remap_term(
        term,
        {"a": "q"},
        {"x": w},
        {},
    )
    expected = Op.make(
        "matmul", w, Op.make("mul", Param("q", _t(4)), Const(2))
    )
    assert out == expected

def test_remap_term_declines_unmapped_leaves():
    x = Var("x", _t(4))
    term = Op.make("mul", x, Param("a", _t(4)))
    assert _remap_term(term, {}, {"x": x}, {}) is None  # unmapped Param
    assert _remap_term(term, {"a": "b"}, {}, {}) is None  # unmapped Var
    # a non-term leaf passes through untouched
    assert _remap_term(7, {}, {}, {}) == 7

def test_param_leaves_and_names():
    p1, p2 = Param("a", _t(4)), Param("b", _t(4))
    term = Op.make("add", p1, Op.make("mul", p2, Const(2)))
    assert _term_param_names(term) == {"a", "b"}
    # DAG sharing — the memo caps re-walks
    shared = Op.make("neg", p1)
    assert _term_param_names(Op.make("add", shared, shared)) == {"a"}
    # unhashable container args are walked through
    assert _term_param_names([p1, [p2]]) == {"a", "b"}

def test_cache_entry_prefers_pre_causal_term():
    x = Var("x", _t(4))
    raw = Op.make("sdpa", x, x, x, validate=False)
    folded = Op.make("sdpa", x, x, x, arg5=True, validate=False)
    ir = _mk_ir(raw, [x], [])
    res = _fake_result(
        folded, ir, {"causal_specialized": True, "pre_causal_term": raw}
    )
    entry = _cache_entry(res)
    assert entry["term"] is raw  # the pre-fold form is stored
    res2 = _fake_result(raw, ir)
    assert _cache_entry(res2)["term"] is raw

def test_cache_entry_records_derived_recipes():
    x = Var("x", _t(4))
    derived_name = "w__heads2"
    dedup = Param(derived_name, _t(1, 2, 4))
    # the EXPORTED ir root references the plain leaf; the extracted
    # term swaps in the derived dedup stack.
    ir = _mk_ir(
        Op.make("mul", x, Param("w", _t(4, 4))), [x], [Param("w", _t(4, 4))]
    )
    term = Op.make(
        "mul",
        x,
        Op.make(
            "reshape",
            Op.make("index_select", dedup, dim=0, index=(0, 0)),
            shape=(4, 4),
        ),
    )
    sharing = {
        "ties": [],
        "derived": {
            derived_name: {"base": "w", "heads": 2, "imap": (0, 0)}
        },
    }
    res = _fake_result(term, ir, {"param_sharing": sharing})
    entry = _cache_entry(res)
    assert entry["derived"][derived_name]["base"] == "w"
    # an unrecorded derived name is an unservable recipe (None)
    term2 = Op.make("mul", Param("mystery", _t(4)), x)
    entry2 = _cache_entry(_fake_result(term2, ir, {}))
    assert entry2["derived"] == {"mystery": None}
