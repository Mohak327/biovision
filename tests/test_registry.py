import pytest

from biovision.core.registry import Registry


def test_register_get_and_names():
    registry = Registry("thing")

    @registry.register("b")
    def make_b():
        return "B"

    registry.register("a")(lambda: "A")
    assert registry.get("b") is make_b
    assert registry.names() == ["a", "b"]


def test_duplicate_name_is_rejected():
    registry = Registry("thing")
    registry.register("a")(object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a")(object())


def test_unknown_name_lists_known_names():
    registry = Registry("species")
    registry.register("mouse")(object())
    with pytest.raises(KeyError, match="unknown species 'cat'; registered: mouse"):
        registry.get("cat")
