"""Substrate-pure tests ported from ``graphopt:tests/test_exact_corner.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""


def test_param_bytes_bills_materialised_folds():
    """The billing rule behind the fixed regressions, at unit level:

    * ``concat(W, W)`` — a param-only concat the lowerer materialises —
      stores 2·numel(W): occurrences under a fold are COPIES.
    * ``matmul(B, A)`` folds to the dense product: billed numel(B@A),
      not numel(B)+numel(A).
    * The same weight read by two CONSUMERS (no replication) still
      dedups by name — shared reads are stored once.
    * ``dag_cost`` agrees (it defers to the DAG-indexed sum rather than
      its subtractive per-node decomposition, which would subtract a
      leaf at every ancestor fold and erase the copies).
    """
    from mlopt_core.cost import dag_cost, param_bytes_cost
    from mlopt_core.ir import Op, Param, TensorType, Var

    W = Param("W", TensorType((64, 64)))
    A = Param("A", TensorType((8, 128)))
    B = Param("B", TensorType((128, 8)))
    x = Var("x", TensorType((4, 64)))
    y = Var("y", TensorType((4, 64)))

    cat = Op.make("concat", W, W, dim=0)
    assert param_bytes_cost(cat) == 2 * 64 * 64
    assert dag_cost(cat, param_bytes_cost) == 2 * 64 * 64

    mm = Op.make("matmul", B, A)
    assert param_bytes_cost(mm) == 128 * 128  # the dense product

    # concat over a foldable child bills the whole materialisation once
    assert (
        param_bytes_cost(Op.make("concat", A, mm, dim=0))
        == (8 + 128) * 128
    )

    # shared READ: two consumers of one weight — one storage entry
    shared = Op.make(
        "add", Op.make("linear", x, W), Op.make("linear", y, W)
    )
    assert param_bytes_cost(shared) == 64 * 64
    assert dag_cost(shared, param_bytes_cost) == 64 * 64

    # a leaf inside a fold AND read elsewhere is stored twice — once as
    # itself, once inside the materialised copy
    both = Op.make("add", Op.make("linear", x, W), cat)
    assert param_bytes_cost(both) == 3 * 64 * 64
