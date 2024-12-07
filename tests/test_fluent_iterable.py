import itertools
from operator import mul
from typing import Iterable, Any, Callable, Mapping
from unittest.mock import Mock
import more_itertools

import pytest

from pyfluent_iterables import fluent, fluent_of, FluentIterable, FluentMapping


######
# Factory methods
######
@pytest.mark.parametrize(
    "iterable",
    [[1, 2, 3, 4], [], range(1, 5), {"a": 1, "b": 2}.items(), {1, 4, 4, 5}, (1, 2, 3), [None], [i for i in range(3)]],
)
def test_fluent_wraps_in_fluent_iterable(iterable):
    assert_same_elements(fluent(iterable), *iterable)


def test_fluent_returns_fluent_mapping_for_mapping():
    mapping = {"a": 1, "b": 2}
    f = fluent(mapping)
    assert isinstance(f, FluentMapping)
    assert_same_mapping(f, expected=mapping)


def test_fluent_creates_from_generator():
    generator = (i for i in range(3))
    f = fluent(generator)
    assert_same_elements(f, 0, 1, 2)


def test_fluent_creates_from_iterator():
    iterator = iter([1, 2, 3])
    assert_same_elements(fluent(iterator), 1, 2, 3)


def test_fluent_creates_from_infinite_iterable():
    f = fluent(itertools.count(0))
    it = iter(f)
    assert next(it) == 0
    assert next(it) == 1
    assert next(it) == 2


def test_fluent_raises_when_creating_from_non_iterable():
    with pytest.raises(ValueError) as e:
        fluent(1)
    assert str(e.value) == "Given value is not Iterable"


def test_creates_empty():
    assert_same_elements(fluent_of(), *[])


def test_fluent_of_creates_from_elements():
    assert_same_elements(fluent_of(1, 2, 3, 4), 1, 2, 3, 4)


######
# Stateless intermediate operations
######
def test_maps_with_transform():
    f = fluent_of("first", "second").map(str.upper)
    assert_same_elements(f, "FIRST", "SECOND")


def test_filters_with_predicate():
    f = fluent_of("yes1", "NO2", "yes3").filter(str.islower)
    assert_same_elements(f, "yes1", "yes3")


def test_filters_falsy_without_predicate():
    f = fluent_of(1, 0, "a", "", ["x"], [], None, range(1), range(0)).filter()
    assert_same_elements(f, 1, "a", ["x"], range(1))


def test_filters_with_predicate_negation():
    f = fluent_of("yes1", "NO2", "yes3").filter_false(str.islower)
    assert_same_elements(f, "NO2")


def test_filters_truthy_without_negated_predicate():
    f = fluent_of(1, 0, "a", "", ["x"], [], None, range(1), range(0)).filter_false()
    assert_same_elements(f, 0, "", [], None, range(0))


def test_retains_instances_of_type():
    f = fluent_of(1, "a", 2, "b").retain_instances_of(int)
    assert_same_elements(f, 1, 2)


def test_enumerates_with_zero_based_index():
    f = fluent_of("a", "b", "c").enumerate()
    assert_same_elements(f, (0, "a"), (1, "b"), (2, "c"))


def test_enumerates_with_given_index():
    f = fluent_of("a", "b", "c").enumerate(start=10)
    assert_same_elements(f, (10, "a"), (11, "b"), (12, "c"))


def test_zips_with_iterables():
    f = fluent_of(1, 2, 3).zip(["a", "b", "c"], ["X", "Y"])
    assert_same_elements(f, (1, "a", "X"), (2, "b", "Y"))


def test_zips_longest():
    f = fluent_of(1, 2, 3).zip_longest(["a", "b", "c"], ["X", "Y"])
    assert_same_elements(f, (1, "a", "X"), (2, "b", "Y"), (3, "c", None))


def test_zips_longest_with_fill_value():
    f = fluent_of(1, 2, 3).zip_longest(["a", "b", "c"], ["X", "Y"], fillvalue="FILL")
    assert_same_elements(f, (1, "a", "X"), (2, "b", "Y"), (3, "c", "FILL"))


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], []),
        ([1], []),
        ([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)]),
    ],
)
def test_pairwise_zips_with_next(iterable, expected):
    f = fluent(iterable).pairwise()
    assert_same_elements(f, *expected)


def test_pairwise_zips_with_next_for_infinite_iterable():
    f = fluent(itertools.count(start=1)).pairwise()
    it = iter(f)
    assert next(it) == (1, 2)
    assert next(it) == (2, 3)
    assert next(it) == (3, 4)


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], []),
        ([[1, 2], [3, 4], [5, 6]], [1, 2, 3, 4, 5, 6]),
        (iter([iter([1, 2]), iter([3, 4])]), [1, 2, 3, 4]),
    ],
)
def test_flattens_iterable_of_iterables(iterable, expected):
    f = fluent(iterable).flatten()
    assert_same_elements(f, *expected)


def test_raises_if_flattening_non_iterables():
    f = fluent_of(1, 2, 3).flatten()
    with pytest.raises(TypeError):
        list(f)


def test_flat_map_transforms_and_flattens():
    f = fluent_of(1, 2, 3).flat_map(range)
    assert_same_elements(f, 0, 0, 1, 0, 1, 2)


def test_chains_with_other_iterables():
    f = fluent_of(1, 2).chain([3, 4], {}, (i for i in range(5, 7)))
    assert_same_elements(f, 1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    "elements,n,expected",
    [
        ([1, 2, 3], 0, []),
        ([1, 2, 3], 1, [1]),
        ([1, 2, 3], 3, [1, 2, 3]),
        ([1, 2, 3], 4, [1, 2, 3]),
        ([], 1, []),
    ],
)
def test_takes_first_n_elements(elements, n, expected):
    f = fluent(elements).take(n)
    assert_same_elements(f, *expected)


def test_raises_if_taking_negative_number_of_elements():
    with pytest.raises(ValueError):
        fluent_of(1, 2).take(-1)


@pytest.mark.parametrize(
    "elements,expected",
    [
        (["A", "B", "c", "D", "e", "F"], ["A", "B"]),
        (["a", "B", "C"], []),
        (["A", "B"], ["A", "B"]),
    ],
)
def test_takes_elements_while_predicate_is_true(elements, expected):
    f = fluent(elements).takewhile(str.isupper)
    assert_same_elements(f, *expected)


@pytest.mark.parametrize(
    "elements,n,expected",
    [
        ([1, 2, 3], 0, [1, 2, 3]),
        ([1, 2, 3], 1, [2, 3]),
        ([1, 2, 3], 3, []),
        ([1, 2, 3], 4, []),
        ([], 1, []),
    ],
)
def test_drops_first_n_elements(elements, n, expected):
    f = fluent(elements).drop(n)
    assert_same_elements(f, *expected)


def test_raises_if_dropping_negative_number_of_elements():
    with pytest.raises(ValueError):
        fluent_of(1, 2).drop(-1)


@pytest.mark.parametrize(
    "elements,expected",
    [
        (["A", "B", "c", "D", "e", "F"], ["c", "D", "e", "F"]),
        (["a", "B", "C"], ["a", "B", "C"]),
        (["A", "B"], []),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_drops_elements_while_predicate_is_true(elements, expected):
    f = fluent(elements).dropwhile(str.isupper)
    assert_same_elements(f, *expected)


# Test that slicing works both for iterables with and without __getitem__()
@pytest.mark.parametrize("fluent_factory", [lambda: fluent_of(0, 1, 2, 3), lambda: fluent(i for i in range(4))])
@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (None, None, None, [0, 1, 2, 3]),
        (0, 2, None, [0, 1]),
        (0, 4, 2, [0, 2]),
        (None, 2, None, [0, 1]),
        (10, None, None, []),
        (2, 1, None, []),
    ],
)
def test_slices_iterable(fluent_factory, start, stop, step, expected):
    f = fluent_factory().slice(start, stop, step)
    assert_same_elements(f, *expected)


@pytest.mark.parametrize(
    "start, stop, step, expected",
    [
        (-2, None, None, [2, 3]),
        (None, -2, None, [0, 1]),
        (None, None, -1, [3, 2, 1, 0]),
        (-2, -4, -1, [2, 1]),
    ],
)
def test_slices_sliceable_iterable_with_negative_args(start, stop, step, expected):
    f = fluent_of(0, 1, 2, 3).slice(start, stop, step)
    assert_same_elements(f, *expected)


@pytest.mark.parametrize(
    "elements, selectors, expected",
    [
        ([1, 2, 3, 4], [True, 0, False, 1], [1, 4]),
        ([1, 2, 3, 4], [True, False, True], [1, 3]),
        ([1, 2], [True, False, True], [1]),
        ([], [], []),
    ],
)
def test_compress_selected_elements_by_selector(elements, selectors, expected):
    f = fluent(elements).compress(selectors)
    assert_same_elements(f, *expected)


@pytest.mark.parametrize("iterable,expected", [([1], [1]), ([1, 2, 3], [1, 3, 6])])
def test_accumulate(iterable, expected):
    f = fluent(iterable).accumulate(lambda a, b: a + b)
    assert_same_elements(f, *expected)


@pytest.mark.parametrize(
    "iterable,initial,expected", [([], 10, [10]), ([1], 10, [10, 11]), ([1, 2, 3], 10, [10, 11, 13, 16])]
)
def test_accumulate_with_initial(iterable, initial, expected):
    f = fluent(iterable).accumulate(lambda a, b: a + b, initial=initial)
    assert_same_elements(f, *expected)


######
# Stateful intermediate operations
######


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], []),
        ([4, 2, 6, 1], [1, 2, 4, 6]),
        (iter([2, 3, 1]), [1, 2, 3]),
    ],
)
def test_sorts_elements_with_default_order(iterable, expected):
    f = fluent(iterable).sort()
    assert_same_elements(f, *expected)


def test_sorts_elements_with_sorting_key():
    f = fluent(["aaa", "b", "cc"]).sort(key=len, reverse=True)
    assert_same_elements(f, "aaa", "cc", "b")


@pytest.mark.parametrize(
    "elements,expected",
    [
        ([], []),
        ([1, 3, 2, 2, 5, 1, 3, 8], [1, 3, 2, 5, 8]),
        (["z", "b", "a", "a"], ["z", "b", "a"]),
        (iter([2, 3, 1]), [2, 3, 1]),
    ],
)
def test_distinct_returns_distinct_elements_in_order(elements, expected):
    f = fluent(elements).distinct()
    assert_same_elements(f, *expected)


def test_groups_by_given_key_to_dictionary():
    dictionary = fluent_of("aa", "Aa", "BB", "bB").group_by(str.upper)
    assert set(dictionary.keys()) == {"AA", "BB"}
    assert dictionary["AA"] == ["aa", "Aa"]
    assert dictionary["BB"] == ["BB", "bB"]


def test_groups_by_unsorted_data():
    dictionary = fluent_of("aa", "BB", "Aa", "bB").group_by(str.upper)
    assert set(dictionary.keys()) == {"AA", "BB"}
    assert dictionary["AA"] == ["aa", "Aa"]
    assert dictionary["BB"] == ["BB", "bB"]


def test_group_by_values_are_reusable():
    dictionary = fluent_of("aa", "aa").group_by()
    group = dictionary["aa"]
    assert list(group) == ["aa", "aa"]
    assert list(group) == ["aa", "aa"]  # Assert second copy/iteration over group is not empty


def test_group_by_values_ordering_is_stable():
    dictionary = fluent_of("aa", "BB", "Aa", "Aa", "bB", "aa", "Bb").group_by(str.upper)
    assert dictionary["AA"] == ["aa", "Aa", "Aa", "aa"]  # Assert same order as in input
    assert dictionary["BB"] == ["BB", "bB", "Bb"]


def test_groups_by_identity_if_no_key_given():
    dictionary = fluent_of(0, 0.0, 1, 1.0).group_by()

    assert dictionary[0] == [0, 0.0]
    assert dictionary[1] == [1, 1.0]


@pytest.mark.parametrize(
    "iterable, expected",
    [
        (["a", "b", "c"], ["c", "b", "a"]),
        ([], []),
        (iter([1, 2, 3]), [3, 2, 1]),
    ],
)
def test_reversed(iterable, expected):
    f = fluent(iterable).reversed()
    assert_same_elements(f, *expected)


@pytest.mark.parametrize(
    "iterable, n, expected",
    [
        ([1, 2, 3, 4], 2, [[1, 2], [3, 4]]),
        ([], 1, []),
        ([1, 2, 3], 1, [[1], [2], [3]]),
        ([1, 2, 3], 2, [[1, 2], [3]]),
        ([1, 2, 3], 4, [[1, 2, 3]]),
        (iter([1, 2, 3]), 2, [[1, 2], [3]]),
    ],
)
def test_grouped_returns_lists_of_size(iterable, n, expected):
    f = fluent(iterable).grouped(n)
    assert_same_elements(f, *expected)


def test_grouped_can_be_repeated():
    f = fluent([1, 2, 3])
    grouped1 = f.grouped(2)
    grouped2 = f.grouped(2)
    assert_same_elements(grouped1, *grouped2)


def test_returns_randomized_iterable():
    f = fluent([1, 2, 3])
    randomized = f.random()
    assert_same_elements(f, 1, 2, 3)
    assert set(randomized) == {1, 2, 3}


def test_returns_random_sample():
    f = fluent([1, 2, 3, 4])
    sample = f.sample(3)
    assert_same_elements(f, 1, 2, 3, 4)
    assert sample.len() == 3
    assert all(i in f for i in sample)


@pytest.mark.parametrize(
    "elements, expected",
    [
        ([("a", 1), ("b", 2)], {"a": 1, "b": 2}),
        ([("a", 1), ("b", 2), ("a", 3)], {"a": 3, "b": 2}),
    ],
)
def test_to_fluent_dict_creates_fluent_mapping(elements, expected):
    f = fluent(elements).to_fluent_dict()
    assert_same_mapping(f, expected=expected)
    assert isinstance(f, FluentMapping)


def test_apply_transform_with_no_args():
    def transform(it):
        return (x * 2 for x in it)

    f = fluent_of(1, 2, 3).apply_transformation(transform)
    assert_same_elements(f, 2, 4, 6)


@pytest.mark.parametrize(
    "iterable,transform,args,kwargs,expected",
    [
        ([1, 2, 3], itertools.islice, (2,), {}, [1, 2]),
        ([1, 2, 3], itertools.islice, (1, None), {}, [2, 3]),
        ([1, 2, 3], itertools.islice, (0, None, 2), {}, [1, 3]),
        ([1, 2, 3], itertools.accumulate, (), {"initial": 10}, [10, 11, 13, 16]),
    ],
)
def test_apply_transformation_with_args_and_kwargs(iterable, transform, args, kwargs, expected):
    f = fluent(iterable).apply_transformation(transform, *args, **kwargs)
    assert_same_elements(f, *expected)


def test_apply_transformation_with_more_itertools():  # see issue #3
    f = fluent_of(1, 2, 3).apply_transformation(more_itertools.windowed, 2)
    assert_same_elements(f, (1, 2), (2, 3))


@pytest.mark.parametrize(
    "elements,operation",
    [
        (["a", "B", ""], lambda f: f.map(str.upper)),
        (["a", "B", ""], lambda f: f.map(str.upper).map(str.lower)),
        (["a", "B", ""], lambda f: f.filter(str.islower)),
        (["a", "B", ""], lambda f: f.filter_false(str.islower)),
        (["a", "B", ""], lambda f: f.enumerate()),
        (["a", "B", ""], lambda f: f.zip([1, 2, 3])),
        (["a", "B", ""], lambda f: f.zip_longest([1, 2], fillvalue=9)),
        (["a", "B", ""], lambda f: f.pairwise()),
        ([[1, 2], [3, 4]], lambda f: f.flatten()),
        ([1, 2], lambda f: f.flat_map(range)),
        ([1, 2], lambda f: f.chain([3, 4])),
        (["a", "B", ""], lambda f: f.take(2)),
        (["a", "B", ""], lambda f: f.takewhile(str.islower)),
        (["a", "B", ""], lambda f: f.drop(2)),
        (["a", "B", ""], lambda f: f.dropwhile(str.isupper)),
        (["a", "B", ""], lambda f: f.compress([1, 0])),
        (["a", "B", ""], lambda f: f.sort()),
        (["a", "B", ""], lambda f: f.distinct()),
        (["a", "B", ""], lambda f: f.group_by()),
        (["a", "B", ""], lambda f: f.reversed()),
        ([("a", 1), ("b", 2)], lambda f: f.to_fluent_dict()),
        ([1, 2, 3], lambda f: f.accumulate(mul)),
        ([1, 2, 3], lambda f: f.grouped(2)),
        ([1, 2, 3], lambda f: f.random()),
        ([1, 2, 3], lambda f: f.sample(2)),
    ],
)
def test_intermediate_operations_do_not_exhaust_iterable(
    elements: Iterable, operation: Callable[[FluentIterable], FluentIterable]
):
    f = operation(fluent(elements))
    first = list(f)  # consume once
    second = list(f)  # consume for the second time
    assert second  # assert not empty
    assert first == second


######
# Operations with side-effects
######


def test_for_each_executes_side_effect_action_on_elements():
    accumulator = []
    fluent_of(1, 2, 3).for_each(lambda e: accumulator.append(e))
    assert accumulator == [1, 2, 3]


def test_for_self_executes_side_effect_on_fluent_wrapper():
    result = []
    fluent_of(1, 2, 3).for_self(lambda f: result.append(f.len()))
    assert result[0] == 3


@pytest.mark.parametrize(
    "action",
    [
        lambda f: f.for_each(str),
        lambda f: f.for_self(str),
    ],
)
def test_side_effect_operations_return_same_iterable(action):
    before = fluent_of(1, 2, 3)
    after = action(before)
    assert after is before


@pytest.mark.parametrize(
    "operation",
    [
        lambda f: f.for_each(str),
        lambda f: f.for_self(str),
    ],
)
def test_side_effect_operations_do_not_exhaust_iterable(operation: Callable[[FluentIterable], FluentIterable]):
    elements = [1, 2, 3]
    f = fluent(elements)
    operation(f)
    assert_same_elements(f, *elements)


######
# Terminal operations
######
def test_copies_to_new_list():
    source = [1, 2, "c"]
    result = fluent(source).to_list()
    assert result == source
    assert result is not source


def test_copies_to_set():
    result = fluent_of(1, 2, 2, 3).to_set()
    assert result == {1, 2, 3}


def test_copies_to_new_set():
    source = {1, 2, "c"}
    result = fluent(source).to_set()
    assert result == source
    assert result is not source


def test_copies_to_frozenset():
    result = fluent_of(1, 2, 2, 3).to_frozenset()
    assert result == {1, 2, 3}
    assert isinstance(result, frozenset)


def test_copies_to_tuple():
    source = [1, 2, "c"]
    result = fluent(source).to_tuple()
    assert result == (1, 2, "c")


def test_creates_dict_from_pairs():
    result = fluent_of(("a", 1), ("b", 2)).to_dict()
    assert result == {"a": 1, "b": 2}


def test_raises_when_creating_dict_from_non_pairs():
    with pytest.raises(ValueError):
        fluent_of(("a", 1), ("b", 2, 2)).to_dict()


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], True),
        (["a", "b"], True),
        (["a", "B"], False),
        (["A", "B"], False),
    ],
)
def test_all_returns_true_iff_all_elements_match_predicate(iterable, expected):
    result = fluent(iterable).all(str.islower)
    assert result == expected


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], True),
        (["a", 1, range(1)], True),
        (["a", 0, "c"], False),
        (["", 0], False),
    ],
)
def test_all_returns_true_iff_all_elements_are_truthy_without_predicate(iterable, expected):
    result = fluent(iterable).all()
    assert result == expected


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], False),
        (["a", "b"], True),
        (["a", "B"], True),
        (["A", "B"], False),
    ],
)
def test_any_returns_true_iff_any_element_matches_predicate(iterable, expected):
    result = fluent(iterable).any(str.islower)
    assert result == expected


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], False),
        (["a", 1, range(1)], True),
        (["a", 0, "c"], True),
        (["", 0], False),
    ],
)
def test_any_returns_true_iff_any_element_is_truthy_without_predicate(iterable, expected):
    result = fluent(iterable).any()
    assert result == expected


@pytest.mark.parametrize(
    "iterable,is_empty",
    [
        ([], True),
        (iter([]), True),
        (["a", 0], False),
        ({"a": 1}, False),
    ],
)
def test_empty_returns_whether_zero_elements_present(iterable, is_empty):
    result = fluent(iterable).empty()
    assert result == is_empty


@pytest.mark.parametrize(
    "iterable,is_empty",
    [
        ([], True),
        (iter([]), True),
        (["a", 0], False),
        ({"a": 1}, False),
    ],
)
def test_not_empty_returns_whether_element_present(iterable, is_empty):
    result = fluent(iterable).not_empty()
    assert not result == is_empty


@pytest.mark.parametrize(
    "iterable,expected",
    [
        ([], 0),
        (["a"], 1),
        ({"a": 1, "b": 2}, 2),
        ("abc", 3),
        (tuple([1, 2]), 2),
        ({1, 2}, 2),
        (range(2), 2),
        (iter([1, 2, 3]), 3),
    ],
)
def test_len_returns_length(iterable, expected):
    f = fluent(iterable)
    assert f.len() == expected


@pytest.mark.parametrize("iterable,expected", [([], 0), ([1, 2, 3], 6)])
def test_sum_returns_sum(iterable, expected):
    f = fluent(iterable)
    assert f.sum() == expected


@pytest.mark.parametrize("iterable,expected", [([], None), ([2, 1, 3], 1)])
def test_min_returns_smallest(iterable, expected):
    f = fluent(iterable)
    assert f.min() == expected


@pytest.mark.parametrize(
    "iterable,function,key,default,expected",
    [
        ([], FluentIterable.min, str.upper, "def", "def"),
        ([], FluentIterable.max, str.upper, "def", "def"),
        (["a2", "b1", "c3", "d2"], FluentIterable.min, lambda s: s[1], "def", "b1"),
        (["a2", "b1", "c3", "d2"], FluentIterable.max, lambda s: s[1], "def", "c3"),
    ],
)
def test_min_max_apply_key_and_default(iterable, function, key, default, expected):
    f = fluent(iterable)
    assert function(f, key=key, default=default) == expected  # type: ignore


@pytest.mark.parametrize("iterable,expected", [([], None), ([2, 1, 3], 3)])
def test_max_returns_largest(iterable, expected):
    f = fluent(iterable)
    assert f.max() == expected


@pytest.mark.parametrize("iterable,expected", [([1], 1), ([1, 2, 3], 6)])
def test_reduce(iterable, expected):
    f = fluent(iterable)
    assert f.reduce(lambda a, b: a + b) == expected


@pytest.mark.parametrize("iterable,initial,expected", [([], 10, 10), ([1], 10, 11), ([1, 2, 3], 10, 16)])
def test_reduce_with_initial(iterable, initial, expected):
    f = fluent(iterable)
    result = f.reduce(lambda a, b: a + b, initial=initial)
    assert result == expected


def test_joins_to_string():
    f = fluent_of("a", 1, 2)
    assert f.join() == "a, 1, 2"
    assert f.join("-") == "a-1-2"
    assert f.join(",", prefix="(", postfix=")") == "(a,1,2)"
    assert f.join(",", prefix="(", postfix=")", transform=lambda x: str(x).upper()) == "(A,1,2)"


def test_first_returns_first_element_if_present():
    f = fluent_of(1, 2, 3)
    assert f.first() == 1


def test_first_returns_none_if_empty():
    assert fluent_of().first() is None


@pytest.mark.parametrize(
    "elements,operation",
    [
        ([1, 2], lambda f: f.to_list()),
        ([1, 2], lambda f: f.to_set()),
        ([1, 2], lambda f: f.to_frozenset()),
        ([1, 2], lambda f: f.to_tuple()),
        ([("a", 1), ("b", 2)], lambda f: f.to_dict()),
        ([1, 2], lambda f: f.all()),
        ([1, 2], lambda f: f.any()),
        ([1, 2], lambda f: f.len()),
        ([1, 2], lambda f: f.sum()),
        ([1, 2], lambda f: f.min()),
        ([1, 2], lambda f: f.max()),
        ([1, 2], lambda f: f.reduce(int.__mul__)),
        ([1, 2], lambda f: f.join()),
        ([1, 2], lambda f: f.first()),
    ],
)
def test_terminal_operations_do_not_exhaust_iterable(elements, operation: Callable[[FluentIterable], Any]):
    f = fluent(elements)
    operation(f)
    assert list(f) == elements


######
# Standard sequence interface support
######
# the protocol for defining immutable containers: to make an immutable container, you need only define __len__ and __getitem__ (more on these later).
def test_iterable_supports_len():
    f = fluent_of(1, 2, 3)
    assert len(f) == 3
    fluent_factory_wrapper = f.map(str)
    assert len(fluent_factory_wrapper) == 3


def test_iterable_supports_contains():
    f = fluent_of(1, 2, 3)
    assert 2 in f
    assert 9 not in f


######
# Regression tests
######
@pytest.mark.parametrize(
    "elements,operation",
    [
        ([1, 2], lambda f: f.to_list()),
        ([1, 2], lambda f: f.to_set()),
        ([1, 2], lambda f: f.to_frozenset()),
        ([1, 2], lambda f: f.to_tuple()),
        ([("a", 1), ("b", 2)], lambda f: f.to_dict()),
    ],
)
def test_does_not_invoke_map_transform_unnecessarily(elements, operation: Callable[[FluentIterable], Any]):
    transform = Mock(side_effect=lambda x: x)
    f = fluent(elements).map(transform)
    operation(f)
    assert transform.call_count == len(elements)


######
# Helper functions
######
def assert_same_elements(actual: Iterable, *expected):
    assert tuple(iter(actual)) == expected


def assert_same_mapping(actual: Mapping, expected: Mapping):
    assert actual == expected
    assert list(actual.keys()) == list(expected.keys())  # assert same order
