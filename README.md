# pyfluent-iterables - fluent Python collections wrapper

This library provides a thin wrapper over Python containers (collections) and [iterables](https://docs.python.org/3/glossary.html#term-iterable) (including `list`, `tuple`, generators and `dict`) and [mappings](https://docs.python.org/3/glossary.html#term-mapping) (including `dict`) with a fluent API.

Here is a real-world usage example:
    
```python
(fluent(get_all_jobs_from_api())                # Fetch list of CI/CD jobs and wrap result in a fluent iterable
  .map(Job.parse_obj)                           # Parse the response to a domain object
  .group_by(lambda job: job.name.split(" ")[0]) # Group jobs by issue number in name prefix 
  .map_values(JobsSummary)                      # Create a summary object from each job group
  .sort_items(lambda job, stats: stats.failed_rate, reverse=True)
                                                # Sort job summaries by failure rate
  .for_each_item(print))                        # Print job summaries
```

## Usage
The API provides two main wrappers and associated factory functions:

* `fluent(iterable: Iterable)` and `fluent_of(*args)` return a `FluentIterable` wrapper, which implement the [iterable](https://docs.python.org/3/glossary.html#term-iterable) contract
* `fluent_dict()` creates a `FluentMapping` wrapper implementing both the [mapping](https://docs.python.org/3/glossary.html#term-mapping) and iterable contracts

Each wrapper then provides methods for transforming the contained elements. Methods follow the [fluent API](https://www.martinfowler.com/bliki/FluentInterface.html) style so that calls can be chained and IDE can effectively automatically suggest methods. Since Python doesn't allow multiline chaining of methods without escaping newlines, the recommended approach for longer chain expressions is to wrap them in parentheses:

```python
result = (fluent_of(1,2,3)
          .map(...)
          ...
         .max())
```

The API largely mirrors the standard library, e.g., collection builtins (`map()`, `any()`, ...), `itertools` and `functools` packages.
It also provides conversion methods to other collections (`to_list()`, `to_tuple()`, `to_dict()`, ...), convenience methods common for functional-style programming (`flat_map()`, `flatten()`), and methods for including side-effects in the call chain (`for_each()`).
`FluentMapping`  provides additional methods relevant for dictionaries (`filter_keys()`, `map_values()`, ...).


### Installation
The library is available at [pypi.org](https://pypi.org/project/pyfluent-iterables/). It can be installed, e.g., with `pip install pyfluent-iterables`.

### Factory methods
Here are some examples of using the factory methods. Note that they can also be conveniently used for creating an equivalent of collection literals. 

```python
# Fluent iterable API for a collection, an iterable, or a generator
fluent([1,2,3])            # FluentIterable wrapping a list
fluent((1,2,3))            # FluentIterable wrapping a tuple
generator = (i for i in range(3))
fluent(generator)          # FluentIterable wrapping a generator
fluent(itertools.count(0)) # FluentIterable wrapping an infinite iterable 

# Fluent iterable API from given elements
fluent_of()                # empty FluentIterable
fluent_of(1, 2, 3, 4)      # FluentIterable wrapping [1, 2, 3, 4] list

# Fluent Mapping API 
fluent({'a': 1, 'b': 2})   # FluentMapping wrapping a dictionary
fluent_dict({'a': 1})      # FluentMapping wrapping a dictionary
fluent_dict({'a': 1}, b=2) # FluentMapping combining a dictionary and explicitly given kwargs
fluent_dict(a=1, b=2) # FluentMapping from given kwargs
fluent_dict([("a", 1), ("b", 2)]) # FluentMapping from a list of (key, value) pairs
```

### Compatibility with standard containers
Both FluentIterable and FluentMapping support standard immutable container contracts with one exception: FluentIterable isn't subscriptable (with `[start:stop:step]`) yet; use the `slice()` method instead.

```python
len(fluent_of(1,2,3))       # 3
2 in fluent_of(1,2,3)       # True
fluent_dict(a=1, b=2)['b']  # 2

fluent_of(1,2,3)[1:2]           # Doesn't work
fluent_of(1,2,3).slice(1,2)     # [2]
fluent_of(1,2,3).to_list()[1:2] # also [2]
```

## Motivation
Python provides list and dictionary comprehensions out of the box. Relying on them is probably the most idiomatic for collection manipulation in Python.
However, more complex operations expressed in comprehensions and standard library modules can be tough to read. Here is the same functionality as above expressed with pure Python: 

```python
jobs = [Job.parse_obj(job) for job in get_all_jobs_from_api]   # Fetch list of CI/CD jobs and parse the response to a domain object
jobs.sort(key=lambda job: job.name)                            # itertools.groupby() requires items to be sorted
jobs_by_issue = itertools.groupby(jobs, key=lambda job: job.name.split(" ")[0])
                                                               # Group jobs by issue number in name prefix
job_summaries = []
for issue, jobs_iter in jobs_by_issue:
    job_summaries.append((issue, JobSummary(list(jobs_iter)))) # Create a summary object from each job group
job_summaries.sort(key=lambda pair: pair[1].failed_rate, reverse=True)
                                                               # Sort job summaries by failure rate
for issue, summary in job_summaries: 
    print(issue, summary)                                      # Print job summaries
```

Judge the readability and convenience of the two implementations for yourself.

Here is a simpler motivating example. Notice the order in which you need to read the pieces of code to follow the execution:

```python
# Python without comprehensions
list(
    map(
        str.upper, 
        sorted(["ab", "cd"], reverse=True)))

# Python with comprehensions
[each.upper()
    for each 
    in sorted(["ab", "cd"], reverse=True)]

# pyfluent-iterables
(fluent_of("ab", "cd")
    .sorted(reverse=True)
    .map(str.upper)
    .to_list())
```

While the last option may be a little longer, it is arguably the most readable. Not the least because it's the only version you can read from beggining to end: the first version needs to be read from right (`sorted`) to left (`list`), the second needs to be read from `for` right and then return to `each.upper()` at the beginning.


Advantages of _pyfluent-iterables_ over vanilla Python include:
* **Improved readability.** Flow of execution can be read from start to beginning (unlike comprehensions which need to be read from the middle to end, and then return to the expression at the start).
* **Better suggestions from IDE.** Rather than remembering what is the full name of a grouping function, one can just select from the methods available on `FluentIterable`.
* **More idiomatic names** common in functional programming. e.g., `fluent(list_of_lists).flatten()` instead of `itertools.chain.from_iterable(list_of_lists)`.

### Related work
Similar libraries already exist, such as [fluentpy](https://github.com/dwt/fluent). However, while pyfluent-iterables focus entirely on a rich interface for standard collections,
_fluentpy_ has broader ambitions which, unfortunately, make it harder to learn and use, and make its usage viral (explicit unwrapping is required). Here are some examples from its documentation: 

```python
# Examples from fluentpy library for comparison
_(range(3)).map(_(dict).curry(id=_, delay=0)._)._   
lines = _.lib.sys.stdin.readlines()._
```

## Design principles
* **Prioritize readability**. Principle of the least surprise; the reader should be able to understand the meaning of code without any prior knowledge.
* **Follow existing conventions** where applicable. Use the same function and parameter names as the standard library and keep the contract of standard Python iterables, list, and dictionaries wherever possible.
* **Maximize developer productivity**. The code should be easy to write in a modern IDE with hints. Provide a rich set of higher-level abstractions. Performance is secondary to productivity (use a specialized library if performance is your focus).
* **Minimize overhead**. Despite the previous point, the library in most cases does not add any per-element processing overhead over the standard library. An extra object or two may be created only while __constructing the iterable chain__, i.e., the overhead is 𝒪(1).
* **No dependencies**. The library does not require any transitive dependencies.  

A chain of operations on `FluentIterable` and `FluentMapping` starts with a source (list, iterator, generator, ...), zero or more intermediate operations, and optionally a terminal operation of an operation with side effects.

Intermediate operations can be stateless (these are **evaluated lazily** when needed) or stateful (these create a copy of the underlying collection). 

Operations with side effects return the underlying collection without any changes after side effect is executed. They are provided just for convenience and can be useful, e.g., for debugging (e.g., `.for_each(print)` can be inserted anywhere in the call chain).

## Overview of methods
Here is an overview of library function and method signatures. See documentation of each method for more details.

Factory functions: 
* `fluent(list_or_iterable)`, `fluent_of(*args)`
* `fluent_dict(mapping_or_iterable, **kwargs)`

Stateless intermediate operations on iterables:
* `map(transform: Callable[[T], R]) -> FluentIterable[R]"`
* `filter(predicate: Optional[Callable[[T], Any]] = None)-> FluentIterable[T]`
* `retain_instances_of(accepted_type: type) -> FluentIterable[T]`
* `filter_false(predicate: Optional[Callable[[T], Any]] = None) -> FluentIterable[T]`
* `enumerate(start: int = 0) -> FluentIterable[Tuple[int,T]]`
* `zip(*with_iterables: Iterable) ->FluentIterable[Tuple]`
* `zip_longest(*with_iterables: Iterable, fillvalue=None) -> FluentIterable[Tuple]`
* `pairwise() -> FluentIterable[Tuple[T, T]]`
* `flatten() -> FluentIterable`
* `flat_map(transform: Callable[[T], Iterable[R]]) -> FluentIterable[R]`
* `chain(*iterables: Iterable) -> FluentIterable`
* `take(n: int) -> FluentIterable[T]`
* `takewhile(predicate: Callable[[T], Any]) -> FluentIterable[T]`
* `drop(n: int) -> FluentIterable[T]`
* `dropwhile(predicate: Callable[[T], Any]) -> FluentIterable[T]`
* `slice(start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> FluentIterable[T]`
* `compress(selectors: Iterable[Any]) -> FluentIterable[T]`
* `accumulate(function: Callable[[Union[T, R], T], R], initial: Optional[R] = None) -> FluentIterable[R]`

Stateful intermediate operations on iterables:
* `sort(key: Callable[[T], Any] = None, reverse: bool = False) -> FluentIterable[T]`
* `distinct() -> FluentIterable[T]`
* `group_by(key: Callable[[T], S] = None) -> FluentMapping[S, List[T]]`
* `reversed() -> FluentIterable[T]`
* `grouped(n: int) -> FluentIterable[List[T]]`
* `groups_generator()`
* `random() -> FluentIterable[T]`
* `sample(k: int) -> FluentIterable[T]`
* `to_fluent_dict() -> FluentMapping"`
* `apply_transformation(self, transformation: Callable[Concatenate[Iterable[T], P], Iterator[R]],  *args: P.args, **kwargs: P.kwargs) -> "FluentIterable[R]"`

Operations with side effects on iterables:
* `for_each(action: Callable[[T], Any]) -> FluentIterable[T]`
* `for_self(action: Callable[[FluentIterable[T]], Any]) -> FluentIterable[T]`

Terminal operations on iterables:
* `to_list() -> List[T]`
* `to_set() -> Set[T]`
* `to_frozenset() -> FrozenSet[T]`
* `to_tuple() -> Tuple[T, ...]`
* `to_dict() -> Dict`
* `join(separator: str = ", ", prefix: str = "", postfix: str = "", transform: Callable[[T], str] = str)`
* `all(predicate: Optional[Callable[[T], bool]] = None) -> bool`
* `any(predicate: Optional[Callable[[T], bool]] = None) -> bool`
* `empty() -> bool`
* `not_empty() -> bool`
* `len() -> int`
* `sum()`
* `min(key: Optional[Callable[[T], Any]] = None, default: Optional[T] = None)`
* `max(key: Optional[Callable[[T], Any]] = None, default: Optional[T] = None)`
* `reduce(function: Callable[[Union[S, T, R], T], R], initial: Optional[S] = _sentinel) -> R`
* `first() -> Optional[T]`

Stateful intermediate operations on dictionaries/mappings:
* `filter_keys(predicate: Optional[Callable[[K], Any]] = None) -> FluentMapping[K,V]`
* `filter_values(predicate: Optional[Callable[[V], Any]] = None) -> FluentMapping[K,V]`
* `map_keys(transform: Callable[[K], R]) -> FluentMapping[R, V]`
* `map_values(transform: Callable[[V], R]) -> FluentMapping[K, R]`
* `map_items(transform: Callable[[K, V], R]) -> FluentMapping[K, R]`
* `sort_items(key: Callable[[K, V], Any], reverse: bool = False) -> FluentMapping[K, R]`

Operations with side effects on dictionaries/mappings:
* `for_each_item(action: Callable[[K, V], Any]) -> FluentMapping[K,V]`
* `for_self(action: Callable[[FluentMapping[K,V]], Any]) -> FluentMapping[K,V]`

### Extensibility
The library implements the most commonly used operations, however, it is not intended to be exhaustive. 
Methods from richer libraries, such as [more-itertools](https://more-itertools.readthedocs.io/en/stable/), can be used together with pyfluent-iterables using the `apply_transformation()` method: 

```python
(fluent_of(1,2,3)
    .apply_transformation(more_itertools.chunked, 2)
    .to_list()) # Produces [(1, 2), (2, 3)]
```