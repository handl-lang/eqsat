"""Backend registry — where model backends advertise themselves.

Plan 0037 (``graphopt:project/plans/
pl-0033-plan-0037-backend-neutral-model-layer.md``) names the gap this
module closes: ``mlopt_core.ports`` declares the *ports*
(:class:`~mlopt_core.ports.Source` /
:class:`~mlopt_core.ports.Sink` /
:class:`~mlopt_core.ports.Composer` /
:class:`~mlopt_core.ports.Meter` /
:class:`~mlopt_core.ports.DevicePolicy`), and
:class:`~mlopt_core.pipeline.Backend` bundles one backend's four
values — but nothing recorded *which backends exist*.  Torch was the
only row, reached through ``mlopt_torch.backend.TorchBackend``;
``kit_onnx`` proved a second backend satisfies the same ports yet
had nowhere to say so.

The registry is **registration-side lazy**: a backend registers
factories (zero-argument callables) or already-built port values,
never a backend module object.  Importing ``mlopt_core.backends``
populates nothing and imports no tensor library — a backend seats
itself by calling :func:`register_backend` (or
:meth:`BackendRegistry.register`) from *its own* module's explicit
entry point, the same seam ``mlopt_kernel.play.register()`` uses for
domains and ``gamekit.players.registry.register_player`` uses for
players.  ``import`` alone must stay side-effect free: consumers
that want the populated registry call the backend packages'
``register()`` functions at their entry point.

What a registration carries
---------------------------
* ``name`` — the format/target key (``"onnx"``, ``"torch"``, …).
* ``source`` — a :class:`Source` or a zero-arg factory returning
  one (``model -> (IR, leaves)``).
* ``targets`` — a non-empty mapping ``{target_name: Sink | factory}``:
  the sink targets this source format lowers *to*.  ONNX registers
  ``{"onnx": OnnxSink}`` (source and target share a format);
  ``{"onnx": OnnxSink, "torch": TorchSink}`` would be the same
  source delivered through two runtimes.
* ``composer`` / ``meter`` / ``device_policy`` — optional ports;
  ``None`` means the backend does not provide that phase (ONNX
  today: no composer, no meter).
* ``oracle`` — reserved for the plan-0037 *Phase 0*
  ``VerifyOracle`` port, which does not exist yet.  The field is
  typed ``Any`` and currently carries whatever the backend's own
  verify path is (``OnnxSink.verify`` today); a real ``VerifyOracle``
  protocol lands with phase 0.
* ``cost_profile`` — reserved for the phase-2 per-backend cost
  calibration (a ``mlopt_core.profile.TargetProfile``-shaped value
  or factory).  ``None`` means "price with the built-in profile".

Duplicate policy
----------------
``register`` refuses a second registration under an existing name —
a silent overwrite would let a stale plugin shadow the real one.
Pass ``replace=True`` to supersede deliberately.

The registry records *declarations*; it does not validate them by
importing the backend.  Port conformance is checked
presence-level (``isinstance`` on the ``@runtime_checkable``
protocols) at registration, and real behaviour is proven by the
backend's own tests — the same trust boundary ``OpTable.register``
already draws.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlopt_core.ports import (
    Composer,
    Meter,
    Sink,
    Source,
)

if TYPE_CHECKING:
    from mlopt_core.pipeline import Backend

__all__ = [
    "DEFAULT_REGISTRY",
    "BackendPortsMissing",
    "BackendRegistration",
    "BackendRegistry",
    "backend_names",
    "find_backends_by_ops",
    "get_backend",
    "register_backend",
]


class BackendPortsMissing(LookupError):
    """A registration cannot assemble a full ``pipeline.Backend``.

    Raised by :meth:`BackendRegistration.build_backend` naming the
    ports the registration does not provide (e.g. ONNX's missing
    ``composer``/``meter``).  The registration is still usable
    port-by-port — ``Optimizer(source=…, sink=…)`` needs neither —
    the error only fires when a caller asks for the bundled value.
    """


def _resolve(value: Any, port: type, what: str) -> Any:
    """Return the port instance behind *value* (instance or factory).

    Classes are always factories — ``isinstance(OnnxSink, Sink)`` is
    *true* (the protocol members live on the class), but the
    registration meant ``OnnxSink()``.  A non-class value that
    conforms is used as-is; anything else callable is invoked as a
    zero-arg factory.
    """
    if isinstance(value, port) and not inspect.isclass(value):
        return value
    if callable(value):
        built = value()
        if isinstance(built, port):
            return built
        raise TypeError(
            f"{what} factory returned {type(built).__name__}, "
            f"not a {port.__name__}"
        )
    raise TypeError(
        f"{what} is neither a {port.__name__} nor a "
        f"zero-arg factory returning one"
    )


@dataclass(frozen=True)
class BackendRegistration:
    """One backend's advertised surface — the registry's record.

    Frozen: a registration is a declaration, and replacing a field
    means registering a new declaration (``replace=True``), not
    mutating a live one.  ``targets`` is normalised to a read-only
    mapping at construction.
    """

    name: str
    source: Source | Callable[[], Source]
    targets: Mapping[str, Sink | Callable[[], Sink]]
    composer: Composer | Callable[[], Composer] | None = None
    meter: Meter | Callable[[], Meter] | None = None
    device_policy: Any = None
    oracle: Any = None
    cost_profile: Any = None
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the declaration without resolving any factory."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("backend name must be a non-empty string")
        if not self.targets:
            raise ValueError(
                f"backend {self.name!r} declares no sink targets"
            )
        for target, sink in self.targets.items():
            if not isinstance(target, str) or not target:
                raise ValueError(
                    f"backend {self.name!r} has a non-string "
                    f"target name {target!r}"
                )
            if not (isinstance(sink, Sink) or callable(sink)):
                raise TypeError(
                    f"backend {self.name!r} target {target!r}: "
                    f"expected a Sink or a zero-arg factory, got "
                    f"{type(sink).__name__}"
                )
        if not (
            isinstance(self.source, Source) or callable(self.source)
        ):
            raise TypeError(
                f"backend {self.name!r}: source must be a Source "
                f"or a zero-arg factory, got {type(self.source).__name__}"
            )
        for port_name, port, value in (
            ("composer", Composer, self.composer),
            ("meter", Meter, self.meter),
        ):
            if value is not None and not (
                isinstance(value, port) or callable(value)
            ):
                raise TypeError(
                    f"backend {self.name!r}: {port_name} must be a "
                    f"{port.__name__} or a zero-arg factory, got "
                    f"{type(value).__name__}"
                )
        # Freeze the mapping view — the dataclass is already frozen,
        # but a caller-held dict must not mutate the record.
        object.__setattr__(self, "targets", dict(self.targets))
        object.__setattr__(self, "metadata", dict(self.metadata))

    # -- lazy resolution -------------------------------------------------

    def source_instance(self) -> Source:
        """Resolve and return the backend's :class:`Source`."""
        return _resolve(self.source, Source, f"{self.name} source")

    def sink_names(self) -> tuple[str, ...]:
        """Return the declared target names, in declaration order."""
        return tuple(self.targets)

    def sink(self, target: str | None = None) -> Sink:
        """Resolve one sink target.

        ``target=None`` selects the only target when the registration
        declares exactly one (the common case — source and sink share
        a format); with several targets the name is required and an
        unknown name raises :class:`KeyError`.
        """
        if target is None:
            if len(self.targets) != 1:
                raise KeyError(
                    f"backend {self.name!r} declares targets "
                    f"{self.sink_names()}; pass a target name"
                )
            target = next(iter(self.targets))
        try:
            declared = self.targets[target]
        except KeyError:
            raise KeyError(
                f"backend {self.name!r} has no target {target!r}; "
                f"declared: {self.sink_names()}"
            ) from None
        return _resolve(
            declared, Sink, f"{self.name} target {target!r}"
        )

    def composer_instance(self) -> Composer | None:
        """Resolve the composer port, or ``None`` when undeclared."""
        if self.composer is None:
            return None
        return _resolve(
            self.composer, Composer, f"{self.name} composer"
        )

    def meter_instance(self) -> Meter | None:
        """Resolve the meter port, or ``None`` when undeclared."""
        if self.meter is None:
            return None
        return _resolve(self.meter, Meter, f"{self.name} meter")

    def supported_ops(
        self, target: str | None = None
    ) -> frozenset[str]:
        """Union (or per-target) ``supported_ops`` of the sink(s).

        Resolves the factories — this is the *capability* read the
        search's ``backend_cost`` bound wants; it instantiates the
        declared sinks, so call it once and hold the result.
        """
        if target is not None:
            return self.sink(target).supported_ops
        ops: set[str] = set()
        for name in self.targets:
            ops |= self.sink(name).supported_ops
        return frozenset(ops)

    def build_backend(self) -> Backend:
        """Assemble the :class:`~mlopt_core.pipeline.Backend` value.

        Raises :class:`BackendPortsMissing` naming the ports the
        registration does not provide — a backend may legitimately
        advertise only ``source``/``sink`` (ONNX does) and still
        drive ``Optimizer(source=…, sink=…)``; the bundled value
        needs all four ports.
        """
        from mlopt_core.pipeline import Backend

        composer = self.composer_instance()
        meter = self.meter_instance()
        missing = [
            port
            for port, value in (
                ("composer", composer),
                ("meter", meter),
            )
            if value is None
        ]
        if missing:
            raise BackendPortsMissing(
                f"backend {self.name!r} does not provide "
                f"{', '.join(missing)}; use the registration's "
                f"source()/sink() ports individually instead of "
                f"build_backend()"
            )
        # ``missing`` is empty, so both resolved — the asserts restate
        # that for the type checker.
        assert composer is not None and meter is not None
        return Backend(
            source=self.source_instance(),
            sink=self.sink(),
            composer=composer,
            meter=meter,
            device_policy=self.device_policy,
        )


class BackendRegistry:
    """The name -> :class:`BackendRegistration` table.

    Instances are independent — tests build their own; production
    code shares :data:`DEFAULT_REGISTRY`.  The registry stores
    declarations only: nothing here imports a backend's runtime,
    and lookups never resolve factories (resolution is the
    registration's own ``source_instance``/``sink`` methods).
    """

    def __init__(self) -> None:
        """Start empty — backends seat themselves explicitly."""
        self._registrations: dict[str, BackendRegistration] = {}

    def register(
        self,
        registration: BackendRegistration,
        *,
        replace: bool = False,
    ) -> BackendRegistration:
        """Seat *registration*; return it for chaining.

        A second registration under an existing name raises
        ``ValueError`` unless ``replace=True`` — silent shadowing is
        how a stale plugin quietly wins.
        """
        if not isinstance(registration, BackendRegistration):
            raise TypeError(
                "register() expects a BackendRegistration, got "
                f"{type(registration).__name__}"
            )
        if registration.name in self._registrations and not replace:
            raise ValueError(
                f"backend {registration.name!r} is already "
                f"registered; pass replace=True to supersede it"
            )
        self._registrations[registration.name] = registration
        return registration

    def unregister(self, name: str) -> BackendRegistration:
        """Remove and return the registration named *name*."""
        try:
            return self._registrations.pop(name)
        except KeyError:
            raise KeyError(
                f"no backend registered as {name!r}"
            ) from None

    def clear(self) -> None:
        """Drop every registration (test isolation)."""
        self._registrations.clear()

    def get(self, name: str) -> BackendRegistration:
        """Return the registration named *name*; ``KeyError`` absent."""
        try:
            return self._registrations[name]
        except KeyError:
            raise KeyError(
                f"no backend registered as {name!r}"
            ) from None

    def __getitem__(self, name: str) -> BackendRegistration:
        """``registry["onnx"]`` — same as :meth:`get`."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """``"onnx" in registry``."""
        return name in self._registrations

    def __iter__(self) -> Iterable[BackendRegistration]:
        """Iterate registrations in registration order."""
        return iter(self._registrations.values())

    def __len__(self) -> int:
        """Return the number of registered backends."""
        return len(self._registrations)

    def names(self) -> tuple[str, ...]:
        """Return the registered names, in registration order."""
        return tuple(self._registrations)

    def items(self) -> tuple[tuple[str, BackendRegistration], ...]:
        """Return ``(name, registration)`` pairs in order."""
        return tuple(self._registrations.items())

    def find_by_ops(
        self,
        required_ops: Iterable[str],
        *,
        target: str | None = None,
    ) -> list[BackendRegistration]:
        """Return registrations whose sink(s) cover ``required_ops``.

        A registration qualifies when *some* declared target's
        ``supported_ops`` is a superset of ``required_ops`` — the
        capability filter a caller uses to ask "which backends can
        deliver a program using these ops".  ``target`` narrows the
        match to one named target across registrations that declare
        it.  This *resolves* factories (it instantiates the sinks),
        so hold the result rather than re-querying per op.
        """
        required = frozenset(required_ops)
        hits: list[BackendRegistration] = []
        for registration in self._registrations.values():
            names = (
                (target,)
                if target is not None
                else registration.sink_names()
            )
            for sink_name in names:
                try:
                    ops = registration.supported_ops(sink_name)
                except KeyError:
                    continue
                if required <= ops:
                    hits.append(registration)
                    break
        return hits


#: The process-wide registry — empty until a backend's explicit
#: ``register()`` seats it.  Importing this module registers nothing.
DEFAULT_REGISTRY = BackendRegistry()


def register_backend(
    name: str,
    *,
    source: Source | Callable[[], Source],
    targets: Mapping[str, Sink | Callable[[], Sink]],
    composer: Composer | Callable[[], Composer] | None = None,
    meter: Meter | Callable[[], Meter] | None = None,
    device_policy: Any = None,
    oracle: Any = None,
    cost_profile: Any = None,
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
    registry: BackendRegistry | None = None,
    replace: bool = False,
) -> BackendRegistration:
    """Seat a backend on :data:`DEFAULT_REGISTRY` (or *registry*).

    The one-call form a backend's ``register()`` wraps::

        from mlopt_core.backends import register_backend

        def register() -> None:
            register_backend(
                "onnx",
                source=OnnxSource,
                targets={"onnx": OnnxSink},
            )

    Factories (``OnnxSource`` / ``OnnxSink`` themselves are
    zero-arg-callable classes) keep the import side-effect free —
    the registry stores the callable and resolves it only when a
    consumer asks for the port.
    """
    reg = registry if registry is not None else DEFAULT_REGISTRY
    return reg.register(
        BackendRegistration(
            name=name,
            source=source,
            targets=targets,
            composer=composer,
            meter=meter,
            device_policy=device_policy,
            oracle=oracle,
            cost_profile=cost_profile,
            notes=notes,
            metadata=metadata or {},
        ),
        replace=replace,
    )


def get_backend(name: str) -> BackendRegistration:
    """Return the :data:`DEFAULT_REGISTRY` registration named *name*."""
    return DEFAULT_REGISTRY.get(name)


def backend_names() -> tuple[str, ...]:
    """Return the names seated on :data:`DEFAULT_REGISTRY`."""
    return DEFAULT_REGISTRY.names()


def find_backends_by_ops(
    required_ops: Iterable[str], *, target: str | None = None
) -> list[BackendRegistration]:
    """Capability-filter :data:`DEFAULT_REGISTRY` by op coverage."""
    return DEFAULT_REGISTRY.find_by_ops(required_ops, target=target)
