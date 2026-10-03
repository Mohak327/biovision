"""A name-to-object registry, used to select species by name."""
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that stores the decorated object under `name`."""

        def decorator(item: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' is already registered")
            self._items[name] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown {self._kind} '{name}'; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)


species: Registry = Registry("species")
