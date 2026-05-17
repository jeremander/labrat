from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence, Sized
from dataclasses import dataclass
from functools import reduce
import itertools
from math import prod
import operator
from typing import Any, cast

from typing_extensions import Self

from labrat import JSONDict


class Params(Iterable[JSONDict], Sized):
    """Class representing a collection of experimental parameters, each one given by a dict of
    (parameter, value) pairs."""

    @property
    @abstractmethod
    def keys(self) -> set[str]:
        """Gets the set of unique keys (parameter names) among all of the parameter dicts."""

    @classmethod
    def empty(cls) -> Params:
        """Empty set of params."""
        return ParamList([])

    @classmethod
    def single(cls, name: str, values: list[Any]) -> Params:
        """Constructor from a parameter name and a list of values.
        This is essentially the same as Params.grid when only a single parameter is present in the grid."""
        return SingleParamGrid(name, values)

    @classmethod
    def product(cls, *params: Self) -> Params:
        """Constructs the Cartesian product of multiple Params, with the resulting dicts merged together.
        Raises a ValueError if any keys overlap."""
        match params:
            case ():
                return ParamList([{}])
            case (ps,):
                return ps
            case _:
                return ParamProduct(list(params))

    @classmethod
    def union(cls, *params: Self) -> Params:
        """Constructs the union of multiple Params."""
        match params:
            case ():
                return cls.empty()
            case (ps,):
                return ps
            case _:
                return ParamUnion(list(params))

    @classmethod
    def grid(cls, grid: dict[str, list[Any]]) -> Params:
        """Given a dict from parameter names to value lists, constructs a parameter grid."""
        return cls.product(*[SingleParamGrid(key, vals) for (key, vals) in grid.items()])

    def with_trials(self, num_trials: int) -> ParamProduct:
        """Given a number of trials, returns a new Params object which includes a "trial" parameter whose values
        are indices ranging from 0 to (num_trials - 1)."""
        return cast(ParamProduct, self * SingleParamGrid('trial', range(num_trials)))

    def __mul__(self, other: Self) -> Params:
        return type(self).product(self, other)

    def __rmul__(self, other: Self) -> Params:
        return type(other).product(other, self)

    def __add__(self, other: Self) -> Params:
        return type(self).union(self, other)

    def __radd__(self, other: Self) -> Params:
        return type(other).union(other, self)


@dataclass
class ParamList(Params):
    """A concrete list of parameter dicts."""
    params: list[JSONDict]

    @property
    def keys(self) -> set[str]:
        return set.union(*map(set, self.params))

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self) -> Iterator[JSONDict]:
        return iter(self.params)


@dataclass
class SingleParamGrid(Params):
    """A collection of values for a single parameter."""
    key: str
    values: Sequence[Any]

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise ValueError('key must be a string')
        if not isinstance(self.values, Sequence):
            raise ValueError('values must be a sequence')

    @property
    def keys(self) -> set[str]:
        return {self.key}

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[JSONDict]:
        key = self.key
        return ({key: val} for val in self.values)


@dataclass
class ParamProduct(Params):
    """A Cartesian product of Params.
    The param dicts from each component will be merged together during enumeration.
    The set of parameters from each component are required to be disjoint."""
    params: list[Params]

    def __post_init__(self) -> None:
        # enforce that the keys from every component are disjoint
        keys = set()
        for ps in self.params:
            for key in ps.keys:
                if key in keys:
                    raise ValueError(
                        f'cannot make a product of Params with overlapping key {key!r}'
                    )
                keys.add(key)

    @property
    def keys(self) -> set[str]:
        return set.union(*(ps.keys for ps in self.params))

    def __len__(self) -> int:
        return prod(map(len, self.params))

    def __iter__(self) -> Iterator[JSONDict]:
        return (reduce(operator.or_, ps, {}) for ps in itertools.product(*self.params))


@dataclass
class ParamUnion(Params):
    """A union of Params."""
    params: list[Params]

    @property
    def keys(self) -> set[str]:
        return set.union(*(ps.keys for ps in self.params))

    def __len__(self) -> int:
        return sum(map(len, self.params))

    def __iter__(self) -> Iterator[JSONDict]:
        return itertools.chain.from_iterable(self.params)
