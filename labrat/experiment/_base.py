from __future__ import annotations

from abc import ABC, abstractmethod
import bdb
from copy import copy
from dataclasses import MISSING, dataclass, fields, is_dataclass, make_dataclass
from datetime import datetime
from functools import cache
from logging import Logger
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, get_args, get_origin

from fancy_dataclass import JSONDataclass
from fancy_dataclass.sql import SQLDataclass, register
from sqlalchemy import Column, Integer

from labrat import LOGGER, get_logger


T = TypeVar('T')

# how to handle errors in experiments
ErrorMode: TypeAlias = Literal['ignore', 'warn', 'raise']


class Result(JSONDataclass, store_type='off'):
    """A class for experimental results, where the fields can be mapped to a SQL table."""


R = TypeVar('R', bound=Result)


class ResultEntry(SQLDataclass, JSONDataclass, store_type='off'):  # type: ignore[misc]
    """A record containing info about an experiment and a single result."""
    session_id: str
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True)
class ExperimentResult(Generic[R]):
    """Bundle of data returned when running an experiment."""
    # experiment that was run
    experiment: Experiment[R]
    # time the experiment was started
    start_time: datetime
    # time the experiment was completed
    end_time: datetime
    # result of the experiment
    result: R | Exception


@dataclass
class Experiment(JSONDataclass, Generic[R], ABC, store_type='off', allow_extra_fields=False):
    """A type representing some kind of computational experiment.

    The type itself corresponds to a SQL table.

    An instance of an Experiment represents an experimental trial with one particular choice of parameters.
    The result will be stored in the type's SQL table."""

    @classmethod
    def logger(cls) -> Logger:
        """Gets a logger for this particular Experiment subclass."""
        return get_logger(cls.__name__)

    @classmethod
    def result_type(cls) -> type[R]:
        """Gets the Result subclass.
        Infers this from the parameter R of the Experiment[R] class from which this subclass should inherit."""
        # TODO: this is brittle, since there could be multiple generic types!
        # Instead, make it a required ClassVar or something.
        for base in cls.__orig_bases__:  # type: ignore[attr-defined]
            if (origin := get_origin(base)) and issubclass(origin, Experiment):
                return get_args(base)[0]  # type: ignore[no-any-return]
        raise TypeError('Experiment subclass must inherit from Experiment[R] for some Result subclass R')

    @classmethod
    def extra_columns(cls) -> dict[str, Column[Any]]:
        """Gets additional columns to provide to the SQL table which are not included among the dataclass fields."""
        return {}

    @classmethod
    @cache
    def result_entry_type(cls) -> type[ResultEntry]:
        """Creates a custom subclass of ResultEntry that has a sqlalchemy-backed SQL table."""
        flds = []
        for cl in [cls, cls.result_type()]:
            assert is_dataclass(cl)
            for fld in fields(cl):
                has_default = (fld.default is not MISSING) or (fld.default_factory is not MISSING)
                # to preserve order, put a dummy default of None for any mandatory fields
                if not has_default:
                    fld = copy(fld)
                    fld.default = None
                flds.append((fld.name, fld.type, fld))
        dcls = make_dataclass(cls.__name__, flds, bases=(ResultEntry,))
        extra_cols = {
            'id': Column('id', Integer, primary_key=True, autoincrement=True),
            **cls.extra_columns(),
        }
        return cast(type[ResultEntry], register(extra_cols=extra_cols)(dcls))

    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, returning a result of type R."""

    def get_result(
        self,
        error_mode: ErrorMode = 'raise',
        verbosity: int = 0,
        debug: bool = False,
    ) -> ExperimentResult[R]:
        """Runs the experiment, returning an ExperimentResult object."""
        if verbosity >= 2:
            LOGGER.info(str(self))
        try:
            start_time = datetime.now()
            result: R | Exception = self.run()
        except Exception as e:
            if debug:
                import pdb  # noqa: T100
                pdb.post_mortem()
            if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)) or (error_mode == 'raise'):
                raise
            if error_mode == 'warn':
                logger = self.logger()
                logger.error(f'{type(e).__name__}:\n\t{self}\n\tERROR: {e}')
            # store the exception object itself in the result
            result = e
        return ExperimentResult(
            experiment=self,
            start_time=start_time,
            end_time=datetime.now(),
            result=result,
        )
