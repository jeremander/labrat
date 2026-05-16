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
import multiprocessing as mp
import random
from typing import Any, Generic, Literal, Optional, TypeAlias, TypeVar, cast, get_args, get_origin

from fancy_dataclass import JSONDataclass
from fancy_dataclass.sql import DEFAULT_REGISTRY, SQLDataclass, register
from sqlalchemy import Column, Integer, and_, create_engine
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from labrat import LOGGER, get_logger
from labrat.params import Params


T = TypeVar('T')

# how to handle errors in experiments
ErrorMode: TypeAlias = Literal['ignore', 'warn', 'raise']


class Result(JSONDataclass, store_type='off'):
    """A class for experimental results, where the fields can be mapped to a SQL table."""


R = TypeVar('R', bound=Result)


class ResultEntry(SQLDataclass, JSONDataclass, store_type='off'):  # type: ignore[misc]
    """A record containing info about an experiment and a single result."""
    experiment_id: str
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
class Experiment(JSONDataclass, Generic[R], ABC, store_type='off'):
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

    def is_in_database(self, session: Session) -> bool:
        """Determines if an Experiment is already in the database for a given Session.
        This means there is at least one result entry for the experiment."""
        result_entry_cls = self.result_entry_type()
        flt = reduce(
            and_,
            (getattr(result_entry_cls, field.name) == getattr(self, field.name) for field in fields(self)),
        )
        rows = session.query(result_entry_cls).filter(flt)
        with suppress(StopIteration):
            _ = next(iter(rows))
            return True
        return False

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
            if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)) or (error_mode == 'raise'):
                raise
            if error_mode == 'warn':
                logger = self.logger()
                logger.error(f'{type(e).__name__}:\n\t{self}\n\tERROR: {e}')
        # TODO: wrap error info into some object, instead of returning None
        return None


@dataclass(frozen=True)
class ExperimentRunner(Iterable[Experiment[R]], Sized):
    """Main driver for running experiments."""
    params: dict[type[Experiment[R]], Params]  # mapping from experiment class to parameters
    engine: str | Engine  # SQL engine
    verbosity: int = 0  # verbosity level
    error_mode: ErrorMode = 'warn'  # how to handle errors (ignore, warn, raise)
    num_threads: int = 1  # number of threads to use
    chunk_size: int = 1  # number of experiments per chunk
    shuffle: bool = False  # shuffle the experiments
    no_rerun: bool = False  # do not re-run the same experiment if already in the database
    debug: bool = False  # drop into debugger if an error occurs (single-threaded only)

    @cached_property
    def _engine(self) -> Engine:
        """Gets the sqlalchemy Engine storing the result data for all experiments."""
        if isinstance(self.engine, str):
            return create_engine(self.engine)
        return self.engine

    def __iter__(self) -> Iterator[Experiment[R]]:
        for (cls, params) in self.params.items():
            yield from (cls.from_dict(d) for d in params)

    def __len__(self) -> int:
        return sum(map(len, self.params.values()))

    @cached_property
    def result_entry_types(self) -> dict[type[Experiment[R]], type[ResultEntry]]:
        """Gets a mapping from Experiment classes to ResultEntry classes which store both parameters and results."""
        return {cls: cls.result_entry_type() for cls in self.params}

    def create_tables(self) -> None:
        """Creates SQL tables for every experiment."""
        # create all the tables
        for cls in self.params:
            _ = cls.result_entry_type()
        DEFAULT_REGISTRY.metadata.create_all(self._engine)

    def make_session(self) -> Session:
        """Creates a new SQLAlchemy session."""
        return sessionmaker(bind=self._engine)()

    def _get_experiments(self, session: Session) -> list[Experiment[R]]:
        experiments = list(self)
        num_experiments = len(self)
        LOGGER.info(f'Parameters specify {num_experiments:,d} experiments.')
        if self.no_rerun:
            LOGGER.info('Filtering out already-run experiments.')
            # filter out experiments already in the database
            experiments = [experiment for experiment in experiments if not experiment.is_in_database(session)]
            num_filtered_experiments = len(experiments)
            if num_filtered_experiments < num_experiments:
                LOGGER.info(f'Filtered to {num_filtered_experiments} experiments.')
        if self.shuffle:
            # TODO: random seed for shuffling?
            random.shuffle(experiments)
        return experiments

    def _process_experiment_result(self, session: Session, experiment_id: str, result: ExperimentResult[R]) -> None:
        """Processes the results of a single experiment."""
        experiment_type = type(result.experiment)
        result_entry_type = self.result_entry_types[experiment_type]
        # combine experiment and result data into a single table entry
        result_dict = {**result.experiment.to_dict(), **result.result.to_dict()}
        entry = result_entry_type.from_dict(result_dict)  # automatically infers subtype
        entry.experiment_id = experiment_id
        entry.start_time = result.start_time
        entry.end_time = result.end_time
        session.add(entry)
        session.commit()

    def run(self) -> None:
        """Runs all experiments."""
        self.create_tables()
        session = self.make_session()
        experiments = self._get_experiments(session)
        num_experiments = len(experiments)
        LOGGER.info(f'Running {num_experiments:,d} experiments with {self.num_threads} thread(s)...')
        # TODO: refactor this into parallel_map utility function (including tqdm wrapper)
        pool = mp.Pool(self.num_threads)
        if self.num_threads == 1:
            mapper: Any = map
            debug = self.debug
        else:
            mapper = partial(pool.imap_unordered, chunksize=self.chunk_size)
            debug = False
        func = partial(Experiment.get_results, error_mode=self.error_mode, verbosity=self.verbosity, debug=debug)
        all_results: Iterable[ExperimentResult[R]] = mapper(func, experiments)
        # TODO: use milliseconds? Use a hash of the experiment data instead?
        # TODO: call this runner_id instead?
        experiment_id = datetime.now().strftime('%Y%m%d%H%M%S')
        for result in tqdm(all_results, total=num_experiments):
            if result is not None:
                self._process_experiment_result(session, experiment_id, result)
        LOGGER.info('\033[1m' + 'DONE!')
