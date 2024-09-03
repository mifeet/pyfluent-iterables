from copy import copy
from typing import Mapping

import pytest

from pyfluent_iterables import fluent_dict, FluentMapping


######
# Factory methods
######
def test_fluent_dict_wraps_in_fluent_mapping():
    mapping = {"a": "value", "b": 1}
    f = fluent_dict(mapping)
    assert_same_mapping(f, expected=mapping)
    mapping["c"] = 3
    assert f["c"] == 3


def test_fluent_dict_creates_from_copy_of_mapping_and_kwargs():
    mapping = {"a": 1}
    f = fluent_dict(mapping, b=2)
    assert_same_mapping(f, a=1, b=2)
    mapping["c"] = 3
    assert "c" not in f


def test_fluent_dict_creates_from_iterable_and_kwargs():
    f = fluent_dict([("a", 1), ("b", 2)], c=3)
    assert_same_mapping(f, a=1, b=2, c=3)


def test_fluent_dict_raises_when_creating_from_non_iterable():
    with pytest.raises(TypeError) as e:
        fluent_dict(1)


def test_fluent_dict_creates_from_kwargs():
    assert_same_mapping(fluent_dict(b=2, c=3), {"b": 2, "c": 3})


######
# Stateful intermediate operations
######
def test_filters_keys_with_predicate():
    f = fluent_dict({"yes1": 1, "NO2": 2, "yes3": 3}).filter_keys(str.islower)
    assert_same_mapping(f, yes1=1, yes3=3)


def test_filters_falsy_keys_without_predicate():
    f = fluent_dict(
        {
            1: "x",
            0: "x",
            "a": "x",
            None: "x",
        }
    ).filter_keys()
    assert_same_mapping(f, {1: "x", "a": "x"})


def test_filters_values_with_predicate():
    f = fluent_dict({"a": "yes1", "b": "NO2", "c": "yes3"}).filter_values(str.islower)
    assert_same_mapping(f, a="yes1", c="yes3")


def test_filters_falsy_values_without_predicate():
    f = fluent_dict(
        a=1,
        b=0,
        c="x",
        d=[],
    ).filter_values()
    assert_same_mapping(f, a=1, c="x")


def test_maps_keys_with_transform_preserving_last_value():
    f = fluent_dict(first=1, second=2, fIRST=3).map_keys(str.upper)
    assert_same_mapping(f, {"FIRST": 3, "SECOND": 2})


def test_maps_values_with_transform():
    f = fluent_dict(first="a", second="b").map_values(str.upper)
    assert_same_mapping(f, first="A", second="B")


def test_maps_items_with_transform():
    f = fluent_dict(a="1", b="2").map_items(str.__add__)
    assert_same_mapping(f, a="a1", b="b2")


def test_sorts_map_items():
    d = dict(a1="z", a2="z", a10="z", a99="y")
    f = fluent_dict(d).sort_items(lambda k, v: v + k)
    assert dict(f) == d
    assert list(f.keys()) == ["a99", "a1", "a10", "a2"]


@pytest.mark.parametrize(
    "operation",
    [
        lambda f: f.filter_keys(),
        lambda f: f.filter_values(),
        lambda f: f.map_keys(str.upper),
        lambda f: f.map_values(str.upper),
        lambda f: f.map_items(str.__add__),
    ],
)
def test_intermediate_operations_preserve_original_mapping(operation):
    original = {"first": "a", "second": "b", "": ""}
    f = fluent_dict(original)
    operation(f)
    assert_same_mapping(f, original)


######
# Normal Mapping behavior
######
def test_index_access_works():
    f = fluent_dict(a=1, b=2)
    assert f["a"] == 1
    assert "b" in f
    assert "c" not in f


def test_fluent_mapping_is_mutable():
    f = fluent_dict(a=1, b=2)
    f["b"] = 22
    assert f["b"] == 22


######
# Operations with side-effects
######
def test_for_each_item_executes_side_effect_on_items():
    accumulator = {}
    fluent_dict(a=1, b=2).for_each_item(lambda k, v: accumulator.update(**{k: v}))
    assert accumulator == {"a": 1, "b": 2}


def test_for_self_executes_side_effect_on_fluent_mapping():
    result = []
    fluent_dict(a=1, b=2).for_self(lambda f: result.append(f))
    assert result[0].len() == 2
    assert type(result[0]) == FluentMapping


@pytest.mark.parametrize(
    "action",
    [
        lambda f: f.for_each_item(lambda k, v: k),
        lambda f: f.for_self(str),
    ],
)
def test_side_effect_operations_return_same_iterable(action):
    before = fluent_dict(a=1, b=2)
    after = action(before)
    assert after is before


######
# Standard sequence interface support
######
def test_iterable_supports_indexing():
    f = fluent_dict(a=1, b=2)
    assert f["a"] == 1


def test_iterable_supports_len():
    f = fluent_dict(a=1, b=2)
    assert len(f) == 2


def test_iterable_supports_contains():
    f = fluent_dict(a=1, b=2)
    assert "a" in f
    assert "x" not in f


######
# Helper functions
######
def assert_same_mapping(actual: Mapping, expected: Mapping = None, **expected_kwargs):
    expected_combined = dict(copy(expected or {}))
    expected_combined.update(expected_kwargs)
    assert actual == expected_combined
    assert list(actual.keys()) == list(expected_combined.keys())  # assert same order
