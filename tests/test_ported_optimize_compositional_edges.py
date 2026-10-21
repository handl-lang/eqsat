"""Substrate-pure tests ported from ``graphopt:tests/test_optimize_compositional_edges.py``.

These functions assert only on ``mlopt_core`` / ``mlopt_orchestrator``
behaviour (plus stdlib/pytest/hypothesis) — they travelled to the
domain repo in the phase-4 split only because sibling tests in the
same file needed domain packages.  Ported back so the substrate's
own suite exercises them.
"""

import pytest

from mlopt_core.attrs import is_positional_attr

from mlopt_core.ir import Const, Op, Param, TensorType, Var

from mlopt_core.ports import CostFn, Verifier, signature_conforms


class _KwOnlyProto:
    """__call__ with kw-only + **kw params — exercises _probes arms."""

    def __call__(self, x, *, flag=True, **kw):
        return x, flag, kw

class _SubProto(_KwOnlyProto):
    """Subclass inherits __call__ — exercises the mro continue arm."""

class _ReqKwOnlyProto:
    """__call__ with a REQUIRED kw-only param → min_kwargs arm."""

    def __call__(self, x, *, req):
        return x, req

class _NoSigProto:
    """__call__ whose signature can't be inspected → None arm."""

    __call__ = vars  # builtins raise ValueError on signature()

class _NoSelfCallProto:
    """__call__ without a leading 'self' — the params[0]-!=-self arc."""

    __call__ = len  # signature (obj, /) — first param isn't 'self'

class _BareProto:
    """No __call__ anywhere → _port_signature returns None."""

def test_intern_unhashable_args():
    """Unhashable args: intern lookup's TypeError arm fires, then
    __post_init__'s hash raises too — invalid args die at mint."""
    with pytest.raises(TypeError):
        Op.make("custom_op", {"unhashable": 1})
    # dict/list attrs are repr-fallback → still intern
    x = Op.make("custom_op", Const(1.0), meta={"nested": [1]})
    y = Op.make("custom_op", Const(1.0), meta={"nested": [1]})
    assert x is y

def test_is_positional_attr():
    assert is_positional_attr("arg0") and is_positional_attr("arg12")
    assert not is_positional_attr("dim") and not is_positional_attr(3)

def test_rulecache_default_dir_env(monkeypatch):
    from mlopt_core.rulecache import _default_cache_dir

    monkeypatch.delenv("MLOPT_RULECACHE_DIR", raising=False)
    assert "mlopt" in str(_default_cache_dir())
    monkeypatch.setenv("MLOPT_RULECACHE_DIR", "/tmp/mlopt_rc_test")
    assert str(_default_cache_dir()) == "/tmp/mlopt_rc_test"

def test_signature_conforms_internals():
    # non-callables → False
    assert not signature_conforms(object(), CostFn)
    assert not signature_conforms("x", CostFn)
    # real fns bind the port's call
    assert signature_conforms(lambda t, memo=None: 1.0, CostFn)
    assert not signature_conforms(lambda a, b, c: 1.0, Verifier)

    # strict probe requires the full call to bind
    def partial_ok(ref, out):
        return None

    assert signature_conforms(partial_ok, Verifier, strict=False)
    assert not signature_conforms(partial_ok, Verifier, strict=True)

def test_signature_conforms_kw_only_probes():
    def binds_everything(x, flag=True, **kw):
        return 1

    assert signature_conforms(binds_everything, _KwOnlyProto)

def test_signature_conforms_mro_and_edges():
    # subclass without its own __call__ → walks mro, continues
    assert signature_conforms(lambda x, flag=True, **kw: 1, _SubProto)
    # required kw-only in the proto → min_kwargs path
    assert signature_conforms(lambda x, req: 1, _ReqKwOnlyProto)
    # uninspectable port __call__ → permissive True
    assert signature_conforms(lambda *a: 1, _NoSigProto)
    # uninspectable fn → permissive True
    assert signature_conforms(vars, CostFn)

def test_signature_conforms_no_self_sig():
    assert signature_conforms(lambda x: 1, _NoSelfCallProto)

def test_signature_conforms_bare_proto():
    assert signature_conforms(lambda *a: 1, _BareProto)
