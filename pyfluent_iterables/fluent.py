import abc
import itertools
import random
from collections import UserDict
from enum import Enum
from functools import reduce
from typing import TypeVar, Iterable, Callable, Mapping, Optional, Any, List, FrozenSet, Set, Sized, Final, Iterator, Tuple, Dict, Union, Literal

T = TypeVar('T')
S = TypeVar('S')
R = TypeVar('R')
K = TypeVar('K')
V = TypeVar('V')


def fluent(iterable: Iterable[T]) -> "FluentIterable[T]":
    """
    Wraps given Iterable in a new FluentIterable. Returns FluentMapping if the iterable is also a Mapping.
    """
    if isinstance(iterable, Mapping):
        return FluentMapping(iterable)
    if isinstance(iterable, Iterable):
        return FluentIterableWrapper(iterable)
    raise ValueError("Given value is not Iterable")


def fluent_of(*args):
    """
    Creates a FluentIterable from given variadic arguments.
    """
    return FluentIterableWrapper(args)


def fluent_dict(mapping_or_iterable: Union[Mapping[K, V], Iterable[Tuple[K, V]], None] = None, **kwargs) -> "FluentMapping[K,V]":
    """
    Creates a FluentMapping wrapping a dict created from arguments.
    The arguments have the same semantics as the built-in dict() method.
    If only a `mapping_or_iterable` that is an instance of Mapping is given, then the FluentMapping will wrap it directly and function as a view over it,
    otherwise the result will wrap a new dictionary object.
    """
    if not kwargs and isinstance(mapping_or_iterable, Mapping):
        return FluentMapping(mapping_or_iterable)
    if mapping_or_iterable is None:
        return FluentMapping[str, Any](kwargs)  # type: ignore[return-value]
    return FluentMapping(dict(mapping_or_iterable, **kwargs))


class _Sentinel(Enum):
    INITIAL_MISSING = object()


class FluentIterable(abc.ABC, Iterable[T]):
    """
    Wrapper providing fluent API for operations on an underlying Iterable.
    If given an Iterable that can be iterated over repeatedly, operations returning a (Fluent)Iterable can also be iterated over repeatedly.
    While instances of FluentIterable itself are immutable, changes to the underlying wrapped Iterable may be visible.
    """

    @abc.abstractmethod
    def _iterable(self) -> Iterable[T]:
        """Returns a reusable Iterable representing this sequence if reuse is possible, or a one-time Iterator otherwise"""
        pass

    ######
    # Stateless intermediate operations
    ######
    def map(self, transform: Callable[[T], R]) -> "FluentIterable[R]":
        """Returns a FluentIterable containing the results of applying the given transform function to each element in this Iterable."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: map(transform, iterable))

    def filter(self, predicate: Optional[Callable[[T], Any]] = None) -> "FluentIterable[T]":
        """Returns a FluentIterable containing only elements matching the given predicate. If no predicate is given, returns only truthy elements."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: filter(predicate, iterable))

    def retain_instances_of(self, accepted_type: type) -> "FluentIterable[T]":
        """Returns a FluentIterable containing only elements that are instances of `accepted_type` (for which isinstance(element, accepted_type) returns True)."""
        return self.filter(lambda it: isinstance(it, accepted_type))

    def filter_false(self, predicate: Optional[Callable[[T], Any]] = None) -> "FluentIterable[T]":
        """Returns a FluentIterable containing only elements NOT matching the given predicate. If no predicate is given, returns only falsy elements."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.filterfalse(predicate, iterable))

    def enumerate(self, start: int = 0) -> "FluentIterable[Tuple[int, T]]":
        """Returns a FluentIterable over pairs of (index, element) for elements in the original Iterable. Indices start with the value of `start`."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: enumerate(iterable, start=start))

    def zip(self, *with_iterables: Iterable) -> "FluentIterable[Tuple]":
        """Returns a sequence of tuples built from the elements of this iterable and other given iterables with the same index. The resulting Iterable ends as soon as the shortest input Iterable ends."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: zip(iterable, *with_iterables))

    def zip_longest(self, *with_iterables: Iterable, fillvalue=None) -> "FluentIterable[Tuple]":
        """Returns a sequence of tuples built from the elements of this iterable and other given iterables with the same index.
         The resulting Iterable is as long as the longest input, missing values are substituted with `fillvalue`."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.zip_longest(iterable, *with_iterables, fillvalue=fillvalue))

    def pairwise(self) -> "FluentIterable[Tuple[T, T]]":
        """Returns a FluentIterable over successive overlapping pairs taken from this iterable."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: _zip_with_next(iter(iterable)))

    def flatten(self) -> "FluentIterable":
        """Returns a FluentIterable over all elements from all Iterables in this iterable. Raises TypeError if not all elements are Iterables."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.chain.from_iterable(iterable))  # type: ignore[arg-type]

    def flat_map(self, transform: Callable[[T], Iterable[R]]) -> "FluentIterable[R]":
        """Returns a FluentIterable over all elements from results of the given transform function applied to each element of this iterable."""
        return self.map(transform).flatten()

    def chain(self, *iterables: Iterable) -> "FluentIterable":
        """Returns a FluentIterable over elements from this iterable followed by all elements from the first iterable given as argument, until all of the iterables are exhausted."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.chain(iterable, *iterables))

    def take(self, n: int) -> "FluentIterable[T]":
        """Returns a FluentIterable over the first `n` elements of this iterable."""
        if n < 0:
            raise ValueError("n must be greater or equal to 0")
        return self.slice(0, n)

    def takewhile(self, predicate: Callable[[T], Any]) -> "FluentIterable[T]":
        """Return a FluentIterable over successive entries from this iterable as long as the predicate evaluates to true for each entry."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.takewhile(predicate, iterable))

    def drop(self, n: int) -> "FluentIterable[T]":
        """Returns a FluentIterable over elements of this iterable with the first `n` elements skipped."""
        if n < 0:
            raise ValueError("n must be greater or equal to 0")
        return self.slice(n)

    def dropwhile(self, predicate: Callable[[T], Any]) -> "FluentIterable[T]":
        """Return a FluentIterable over elements of this iterable that drops elements from this as long as the predicate is true; afterwards, returns every element."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.dropwhile(predicate, iterable))

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> "FluentIterable[T]":
        """
        Returns a FluentIterable over selected elements from this iterable.
        The parameters have the same semantics as for Python list slicing (see also the `slice()` function) or for `itertool.islice()`.
        If the underlying Iterable does not support the `__getitem__()` method, negative values for start, stop, or step are not supported.
        """
        iterable = self._iterable()
        if hasattr(iterable, "__getitem__"):
            return FluentIterableWrapper(iterable[slice(start, stop, step)])  # type: ignore[index]
        return FluentFactoryWrapper(lambda: itertools.islice(iterable, start, stop, step))

    def compress(self, selectors: Iterable[Any]) -> "FluentIterable[T]":
        """Returns a FluentIterable that filters elements from this iterable returning only those that have a corresponding element in `selectors` that evaluates to True.
         Stops when either the data or selectors iterables has been exhausted."""
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.compress(iterable, selectors))

    def accumulate(self, function: Callable[[Union[T, R], T], R], initial: Optional[R] = None) -> "FluentIterable[R]":
        """
        Returns a FluentIterable of accumulated results of a binary function. The iterable starts with `default`, if given,
        or the first element otherwise, and continues by applying `function` to current accumulator value and subsequent elements of this iterable.
        E.g., [1,2,3].accumulate(lambda x, y: x+y, 0) produces 0, 1, 3, 6.
        """
        iterable = self._iterable()
        return FluentFactoryWrapper(lambda: itertools.accumulate(iterable, function, initial=initial))

    ######
    # Stateful intermediate operations
    ######
    def sort(self, key: Callable[[T], Any] = None, reverse: bool = False) -> "FluentIterable[T]":
        """
        Returns a FluentIterable containing elements from this iterable sorted by the output of the `key` function applied to all elements of this iterable
        (identity is used if `key` is None). Returns elements in the reverse order if `reverse` is True.
        This operation is stateful.
        """
        return FluentIterableWrapper(sorted(self._iterable(), key=key, reverse=reverse))  # type: ignore[type-var, arg-type]

    def distinct(self) -> "FluentIterable[T]":
        """Returns iterable over distinct elements in the underlying iterable, preserving the original order"""
        # Dictionary is guaranteed to preserve insertion order since 3.7
        return FluentIterableWrapper(dict.fromkeys(self._iterable()))

    def group_by(self, key: Callable[[T], S] = None) -> "FluentMapping[S, List[T]]":
        """
        Groups elements of this iterable by values of the `key` function applied to each element (the element itself is used if `key` is not provided).
        Returns a FluentMapping where the key corresponds to the result of the `key` function and value corresponds to a list of all elements which mapped to the respective key.
        This operation is stateful.
        """
        # Use a custom implementation to avoid sorting first required by itertools
        key = key or _identity
        result: Dict[S, List[T]] = {}
        for element in self:
            element_key = key(element)
            group = result.get(element_key, [])
            if not group:
                result[element_key] = group
            group.append(element)
        return FluentMapping(result)

    def reversed(self) -> "FluentIterable[T]":
        """
        Returns FluentIterable over elements of this iterable in reversed order.
        This operation can be stateful - creates a copy of elements in the iterable if `__reversed__()` is not available on the underlying iterable.
        """
        iterable = self._iterable()
        if hasattr(iterable, "__reversed__"):
            return FluentFactoryWrapper(lambda: iterable.__reversed__())  # type: ignore[attr-defined]
        else:
            copy = list(iterable)
            return FluentFactoryWrapper(lambda: copy.__reversed__())

    def grouped(self, n: int) -> "FluentIterable[List[T]]":
        """
        Returns iterable over lists of consecutive elements from this iterable.
        Each lists has size `n` unless there are less than `n` elements remaining in which case only the remaining elements are contained in the last list.
        """
        iterable = self._iterable()

        def groups_generator():
            iterator = iter(iterable)
            while True:
                next_slice = list(itertools.islice(iterator, n))
                if not next_slice:
                    return
                yield next_slice

        return FluentFactoryWrapper(groups_generator)

    def random(self) -> "FluentIterable[T]":
        """
        Returns iterable with the same elements as this iterable but with a randomized order (using `random.shuffle()`).
        """
        population = list(self)
        random.shuffle(population)
        return FluentIterableWrapper(population)

    def sample(self, k: int) -> "FluentIterable[T]":
        """
        Returns iterable with a random sample of k elements from this iterable (using `random.sample()`).
        """
        population = list(self)
        return FluentIterableWrapper(random.sample(population, k))

    def to_fluent_dict(self) -> "FluentMapping":
        """
        Creates a FluentMapping from iterable of pairs.
        The first object of each item becomes a key in the new dictionary, and the second object the corresponding value.
        If a key occurs more than once, the last value for that key is used.
        This operation is stateful.
        """
        return FluentMapping(self.to_dict())

    ######
    # Operations with side-effects
    ######
    def for_each(self, action: Callable[[T], Any]) -> "FluentIterable[T]":
        """Executes the given action on each element and returns self."""
        for e in self:
            action(e)
        return self

    def for_self(self, action: Callable[["FluentIterable[T]"], Any]) -> "FluentIterable[T]":
        """Executes the given action with self as the argument and returns this FluentIterable."""
        action(self)
        return self

    ######
    # Terminal operations
    ######
    def to_list(self) -> List[T]:
        """Returns a new list containing elements of this iterable"""
        return list(self._iterable())

    def to_set(self) -> Set[T]:
        """Returns a new set containing elements of this iterable"""
        return set(self._iterable())

    def to_frozenset(self) -> FrozenSet[T]:
        """Returns a new frozenset containing elements of this iterable"""
        return frozenset(self._iterable())

    def to_tuple(self) -> Tuple[T, ...]:
        """Returns a new tuple containing elements of this iterable"""
        return tuple(self._iterable())

    def to_dict(self) -> Dict:
        """
        Creates a dict from iterable of pairs.
        The first object of each item becomes a key in the new dictionary, and the second object the corresponding value.
        If a key occurs more than once, the last value for that key is used.
        """
        return dict(self._iterable())  # type: ignore[arg-type]

    def join(self, separator: str = ", ", prefix: str = "", postfix: str = "", transform: Callable[[T], str] = str):
        """
        Return string concatenation of all elements in this iterable, separated by `separator`, and optionally prefixed/postfixed with `prefix`/`postfix`, respectively.
        The `transform` function is applied to each element before concatenating.
        """
        return prefix + separator.join(self.map(transform)._iterable()) + postfix

    def all(self, predicate: Optional[Callable[[T], bool]] = None) -> bool:
        """
        If predicate is given, returns True if the result of applying predicate to every element evaluates to true.
        If predicate is not given, returns True if all elements in this iterable evaluate to true.
        If the iterable is empty, returns True.
        """
        if predicate is not None:
            return all(self.map(predicate))
        else:
            return all(self)

    def any(self, predicate: Optional[Callable[[T], bool]] = None) -> bool:
        """
        If predicate is given, returns True if this iterable contains an element such that applying predicate to it evaluates to True.
        If predicate is not given, returns True if any element in this iterable evaluates to true.
        """
        if predicate is not None:
            return any(self.map(predicate))
        else:
            return any(self)

    def empty(self) -> bool:
        """Returns true if and only if this iterable does not contain any elements.
         Note that if the underlying iterable cannot be consumed repeatedly, this method can consume the first element."""
        return not self.not_empty()

    def not_empty(self) -> bool:
        """Returns true if and only if this iterable contains at least one element.
        Note that if the underlying iterable cannot be consumed repeatedly, this method can consume the first element."""
        return any(True for _ in self._iterable())

    def len(self) -> int:
        """Returns the number of elements in this iterable"""
        it = self._iterable()
        if isinstance(it, Sized):
            return len(it)
        else:
            count = 0
            for _ in it:
                count += 1
            return count

    def sum(self):
        """Returns the sum of elements in this iterable with the sum() built-in function"""
        return sum(self._iterable())

    def min(self, key: Optional[Callable[[T], Any]] = None, default: Optional[T] = None):
        """
        Return the smallest item in this iterable. The arguments have identical meaning to the min() built-in function:
        `key` specifies a function used to extract a comparison key, `default` specifies result value if this iterable is empty.
        """
        return min(self._iterable(), key=key, default=default)

    def max(self, key: Optional[Callable[[T], Any]] = None, default: Optional[T] = None):
        """
        Return the smallest item in this iterable. The arguments have identical meaning to the min() built-in function:
        `key` specifies a function used to extract a comparison key, `default` specifies result value if this iterable is empty.
        """
        return max(self._iterable(), key=key, default=default)

    def reduce(self, function: Callable[[Union[S, T, R], T], R], initial: Union[S, None, Literal[_Sentinel.INITIAL_MISSING]] = _Sentinel.INITIAL_MISSING) -> R:
        """
        Accumulates value starting with `default`, if given, or the first element otherwise, and applying `function` from to current accumulator value and each element of this iterable.
        E.g., [1,2,3].reduce(lambda x, y: x+y, 0) calculates (((0+1)+2)+3).
        """
        if initial is _Sentinel.INITIAL_MISSING:
            return reduce(function, self._iterable())  # type: ignore
        return reduce(function, self._iterable(), initial)  # type: ignore

    def first(self) -> Optional[T]:
        """Returns first element of this iterable or None if it is empty."""
        try:
            return next(self.__iter__())
        except StopIteration:
            return None


class FluentIterableWrapper(FluentIterable[T]):
    """Implementation that wraps an existing reusable Iterable"""
    inner: Final[Iterable]

    def __init__(self, iterable: Iterable[T]):
        self.inner = iterable

    def __iter__(self) -> Iterator[T]:
        return self.inner.__iter__()

    def _iterable(self):
        return self.inner


class FluentFactoryWrapper(FluentIterable[T]):
    """Implementation for cases where a known factory to a non-reusable Iterator is available"""

    def __init__(self, factory: Callable[[], Iterator[T]]):
        self._factory = factory

    def __iter__(self) -> Iterator[T]:
        return self._factory()

    def _iterable(self):
        return self


class FluentMapping(UserDict, Mapping[K, V], FluentIterable[K]):
    """
    Wrapper providing fluent API for operations on the underlying Mapping (dict).
    This class also implements FluentIterable over keys of the mapping.
    While instances of FluentMapping itself are immutable, changes to the underlying wrapped Mapping (dict) may be visible.
    """

    def __init__(self, initial: Mapping[K, V]):
        super().__init__()
        self.data = initial  # type: ignore

    def _iterable(self):
        return self.data

    ######
    # Map-specific stateful intermediate operations
    ######
    def filter_keys(self, predicate: Optional[Callable[[K], Any]] = None) -> "FluentMapping[K,V]":
        """Returns a FluentMapping that wraps dictionary created from this mapping by omitting items
         for which applying `predicate` to the key evaluates to false, or for which the key itself evaluates to false if predicate is not given."""
        predicate = predicate or _identity
        return FluentMapping({k: v for k, v in self.items() if predicate(k)})

    def filter_values(self, predicate: Optional[Callable[[V], Any]] = None) -> "FluentMapping[K,V]":
        """Returns a FluentMapping that wraps dictionary created from this mapping by omitting items
         for which applying `predicate` to the value evaluates to false, or for which the value itself evaluates to false if predicate is not given."""
        predicate = predicate or _identity
        return FluentMapping({k: v for k, v in self.items() if predicate(v)})

    def map_keys(self, transform: Callable[[K], R]) -> "FluentMapping[R, V]":
        """Returns a FluentMapping created from this mapping by replacing keys with result of applying `transform` to each key."""
        return FluentMapping({transform(k): v for k, v in self.items()})

    def map_values(self, transform: Callable[[V], R]) -> "FluentMapping[K, R]":
        """Returns a FluentMapping created from this mapping by replacing values with result of applying `transform` to each value."""
        return FluentMapping({k: transform(v) for k, v in self.items()})

    def map_items(self, transform: Callable[[K, V], R]) -> "FluentMapping[K, R]":
        """Returns a FluentMapping created from this mapping by replacing values with result of applying `transform` to each (key, value) pair."""
        return FluentMapping({k: transform(k, v) for k, v in self.items()})

    def sort_items(self, key: Callable[[K, V], Any], reverse: bool = False) -> "FluentMapping[K, R]":
        """Returns a FluentMapping with the same (key, value) pairs but which has its iterating order determined by the result of the `key` function applied to each pair."""
        # Note that dictionaries preserve insertion order (guaranteed since 3.7, CPython implementation detail before)
        return FluentMapping({k: self.data[k] for k, v in sorted(self.items(), key=lambda kv_pair: key(*kv_pair), reverse=reverse)})

    ######
    # Operations with side-effects
    ######
    def for_each_item(self, action: Callable[[K, V], Any]) -> "FluentMapping[K,V]":
        """Executes the given action on each key, value pair and returns self."""
        for key, value in self.items():
            action(key, value)
        return self

    def for_self(self, action: Callable[["FluentMapping[K,V]"], Any]) -> "FluentMapping[K,V]":
        """Executes the given action with self as the argument and returns this FluentMapping."""
        action(self)
        return self


def _identity(it):
    return it


def _zip_with_next(iterator: Iterator[T]) -> Iterator[Tuple[T, T]]:
    try:
        current_element = next(iterator)
    except StopIteration:
        return
    try:
        while True:
            next_element = next(iterator)
            yield current_element, next_element
            current_element = next_element
    except StopIteration:
        return
