from __future__ import annotations

from abc import ABC, abstractmethod
import bdb
from collections.abc import Iterable
from contextlib import suppress
from copy import copy
from dataclasses import MISSING, dataclass, fields, is_dataclass, make_dataclass
from datetime import datetime
from functools import cache, cached_property, reduce
from logging import Logger
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast, get_args, get_origin

from fancy_dataclass import JSONDataclass
from fancy_dataclass.sql import DEFAULT_REGISTRY, SQLDataclass, register
from sqlalchemy import Column, Integer, and_, create_engine
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

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


class ExperimentResultWriter(ABC):
    """Class responsible for writing an ExperimentResult to a file or database."""

    def setup(self, experiment_types: Iterable[type[Experiment[Any]]]) -> None:  # noqa: B027
        """Given the Experiment types which will be run, performs any setup steps, such as creating files or database
        tables."""
        pass

    @abstractmethod
    def experiment_was_written(self, experiment: Experiment[Any]) -> bool:
        """Determines if an Experiment was already written.
        This means there is at least one result entry corresponding to the experiment."""

    @abstractmethod
    def write_experiment_result(self, session_id: str, result: ExperimentResult[R]) -> None:
        """Writes the result of a single experiment."""


@dataclass(frozen=True)
class SQLExperimentResultWriter(ExperimentResultWriter):
    engine: Engine  # sqlalchemy Engine storing result data for all experiments

    def __init__(self, engine: str | Engine) -> None:
        if isinstance(engine, str):
            engine = create_engine(engine)
        object.__setattr__(self, 'engine', engine)

    @cached_property
    def session(self) -> Session:
        return sessionmaker(bind=self.engine)()

    def setup(self, experiment_types: Iterable[type[Experiment[Any]]]) -> None:
        # creates SQL tables for every experiment
        for experiment_type in experiment_types:
            _ = experiment_type.result_entry_type()
        DEFAULT_REGISTRY.metadata.create_all(self.engine)
        # create the Session object for performing database actions
        _ = self.session

    def experiment_was_written(self, experiment: Experiment[Any]) -> bool:
        # check if any result for this experiment is already in the database
        result_entry_type = experiment.result_entry_type()
        flt = reduce(
            and_,
            (getattr(result_entry_type, field.name) == getattr(experiment, field.name) for field in fields(experiment)),
        )
        rows = self.session.query(result_entry_type).filter(flt)
        with suppress(StopIteration):
            _ = next(iter(rows))
            return True
        return False

    def write_experiment_result(self, session_id: str, result: ExperimentResult[Any]) -> None:
        if isinstance(result.result, Exception):
            # for now, do not write any result
            # TODO: eventually include an error column so errors can be stored
            return
        experiment_type = type(result.experiment)
        result_entry_type = experiment_type.result_entry_type()
        # combine experiment and result data into a single table entry
        result_dict = {**result.experiment.to_dict(), **result.result.to_dict()}
        entry = result_entry_type.from_dict(result_dict)  # automatically infers subtype
        entry.session_id = session_id
        entry.start_time = result.start_time
        entry.end_time = result.end_time
        # write the result to the database
        self.session.add(entry)
        self.session.commit()


def sql_writer(engine: str | Engine) -> SQLExperimentResultWriter:
    """Convenience function for constructing a SQLExperimentResultWriter."""
    return SQLExperimentResultWriter(engine)
