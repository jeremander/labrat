from __future__ import annotations

from abc import ABC, abstractmethod
import bdb
from collections.abc import Iterable, Iterator, Sized
from contextlib import suppress
from copy import copy
from dataclasses import MISSING, dataclass, fields, is_dataclass, make_dataclass
from datetime import datetime
from functools import cache, cached_property, partial, reduce
from logging import Logger
import random
from typing import Any, Generic, Literal, Optional, TypeAlias, TypeVar, cast, get_args, get_origin

from fancy_dataclass import JSONDataclass
from fancy_dataclass.sql import DEFAULT_REGISTRY, SQLDataclass, register
from sqlalchemy import Column, Integer, and_, create_engine
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from labrat import LOGGER, get_logger
from labrat.params import Params
from labrat.utils import parallel_map


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
    result: R


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
                    field = copy(fld)
                    field.default = None
                flds.append((field.name, field.type, field))
        dcls = make_dataclass(cls.__name__, flds, bases=(ResultEntry,))
        extra_cols = {
            'id': Column('id', Integer, primary_key=True, autoincrement=True),
            **cls.extra_columns(),
        }
        return cast(type[ResultEntry], register(extra_cols=extra_cols)(dcls))

    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, returning a result of type R."""

    def get_results(
        self,
        error_mode: ErrorMode = 'raise',
        verbosity: int = 0,
        debug: bool = False,
    ) -> Optional[ExperimentResult[R]]:
        """Runs the experiment, returning an ExperimentResult object."""
        if verbosity >= 2:
            LOGGER.info(str(self))
        try:
            start_time = datetime.now()
            result = self.run()
            return ExperimentResult(
                experiment=self,
                start_time=start_time,
                end_time=datetime.now(),
                result=result,
            )
        except Exception as e:
            if debug:
                exception = e  # noqa: F841
                breakpoint()  # noqa: T100
            if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)) or (error_mode == 'raise'):
                raise
            if error_mode == 'warn':
                logger = self.logger()
                logger.error(f'{type(e).__name__}:\n\t{self}\n\tERROR: {e}')
        # TODO: wrap error info into some object, instead of returning None
        return None


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


@dataclass(frozen=True)
class ExperimentRunner(Iterable[Experiment[Any]], Sized):
    """Main driver for running experiments."""
    params: dict[type[Experiment[Any]], Params]  # mapping from experiment class to parameters
    result_writer: ExperimentResultWriter  # object responsible for writing the results
    verbosity: int = 0  # verbosity level
    error_mode: ErrorMode = 'warn'  # how to handle errors (ignore, warn, raise)
    num_threads: int = 1  # number of threads to use
    chunk_size: int = 1  # number of experiments per chunk
    shuffle: bool = False  # shuffle the experiments
    no_rerun: bool = False  # do not re-run the same experiment if already in the database
    debug: bool = False  # drop into debugger if an error occurs (single-threaded only)

    def __iter__(self) -> Iterator[Experiment[Any]]:
        for (cls, params) in self.params.items():
            yield from (cls.from_dict(d) for d in params)

    def __len__(self) -> int:
        return sum(map(len, self.params.values()))

    @cached_property
    def result_entry_types(self) -> dict[type[Experiment[Any]], type[ResultEntry]]:
        """Gets a mapping from Experiment classes to ResultEntry classes which store both parameters and results."""
        return {cls: cls.result_entry_type() for cls in self.params}

    def _get_experiments(self) -> list[Experiment[Any]]:
        experiments = list(self)
        num_experiments = len(self)
        LOGGER.info(f'Parameters specify {num_experiments:,d} experiments.')
        if self.no_rerun:
            LOGGER.info('Filtering out already-run experiments.')
            # filter out experiments which were already written
            experiments = [
                experiment for experiment in experiments if not self.result_writer.experiment_was_written(experiment)
            ]
            num_filtered_experiments = len(experiments)
            if num_filtered_experiments < num_experiments:
                LOGGER.info(f'Filtered to {num_filtered_experiments} experiments.')
        if self.shuffle:
            # TODO: random seed for shuffling?
            random.shuffle(experiments)
        return experiments

    def run(self) -> None:
        """Runs all experiments."""
        self.result_writer.setup(self.params)
        experiments = self._get_experiments()
        num_experiments = len(experiments)
        LOGGER.info(f'Running {num_experiments:,d} experiments with {self.num_threads} thread(s)...')
        # only enter debugger if in single-threaded mode
        debug = self.debug if (self.num_threads == 1) else False
        func = partial(Experiment.get_results, error_mode=self.error_mode, verbosity=self.verbosity, debug=debug)
        all_results = parallel_map(func, experiments, num_threads=self.num_threads, progress=True)
        # TODO: use milliseconds? Use a hash of the experiment data instead?
        # create ID for the entire session of experiment runs
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        LOGGER.info('Processing results...')
        for result in all_results:
            # TODO: handle errors
            if result is not None:
                self.result_writer.write_experiment_result(session_id, result)
        LOGGER.info('\033[1m' + 'DONE!')
