"""Common base of interchangeable components: typed Settings, build context and Registry.

Adding a new implementation (e.g. another detector):
    1. subclass the matching abstract class (HoofDetector, ClawSegmenter, ...),
    2. add a nested `Settings` dataclass with its parameters,
    3. register it: `@DETECTORS.register("name")`,
    4. create it by name with `REGISTRY.create("name", params, ctx)`.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Mapping, TypeVar

from .exceptions import ConfigError

T = TypeVar("T")


@dataclass(frozen=True)
class BuildContext:
    """Run-time parameters shared by all components."""

    base_dir: Path          # relative paths (weight files) are resolved against this directory
    out_dir: Path


@dataclass(frozen=True)
class Settings:
    """Base class of component settings: unknown or missing keys raise ConfigError."""

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None, base_dir: Path | None = None):
        params = dict(params or {})
        fields = {f.name: f for f in dataclasses.fields(cls)}

        unknown = sorted(set(params) - set(fields))
        if unknown:
            raise ConfigError(f"{cls.__qualname__}: unknown parameter(s): {unknown}, accepted: {sorted(fields)}")

        hints = typing.get_type_hints(cls)
        values: dict[str, Any] = {}
        for name, value in params.items():
            if value is not None and _is_path_type(hints.get(name)):
                value = _as_path(value, base_dir, name)
            values[name] = value

        try:
            return cls(**values)
        except TypeError as exc:  # missing required field
            raise ConfigError(f"{cls.__qualname__}: {exc}") from exc


def _is_path_type(hint: Any) -> bool:
    if hint is Path:
        return True
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        return Path in typing.get_args(hint)
    return False


def _as_path(value: Any, base_dir: Path | None, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(f"'{name}' must be a path, got: {type(value).__name__}")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


class Component:
    """Base of every registrable component. Subclasses override Settings."""

    Settings: ClassVar[type[Settings]] = Settings

    @classmethod
    def from_config(cls, params: Mapping[str, Any] | None, ctx: BuildContext):
        """Default construction: Settings resolved from the parameters. Override if the component needs more."""
        return cls(cls.Settings.from_mapping(params, ctx.base_dir))


class Registry(Generic[T]):
    """Name -> implementation mapping for one component type."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            return cls
        return decorator

    def names(self) -> list[str]:
        return sorted(self._items)

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(f"Unknown {self.kind}: '{name}'. Available: {self.names()}") from None

    def create(self, name: str, params: Mapping[str, Any] | None, ctx: BuildContext) -> T:
        return self.get(name).from_config(params, ctx)  # type: ignore[attr-defined]
