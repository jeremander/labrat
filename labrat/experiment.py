from abc import ABC, abstractmethod
import bdb
from collections.abc import Iterator
from copy import copy
from dataclasses import MISSING, dataclass, make_dataclass
from datetime import datetime
from functools import cache, partial, reduce
from logging import Logger
import multiprocessing as mp
import random
from typing import Any, Generic, Optional, TypedDict, TypeVar

from fancy_dataclass.sql import DEFAULT_REGISTRY, ColumnMap, SQLDataclass, register
from sqlalchemy import Column, DateTime, Integer, String, and_
from sqlalchemy.future.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from labrat import LOGGER, JSONDict, get_logger
from labrat.params import Params


T = TypeVar('T')


class Result(SQLDataclass):
    """A class for experimental results, where the fields can be mapped to a SQL table."""


R = TypeVar('R', bound=Result)


class Experiment(SQLDataclass, Generic[R], ABC):

    @classmethod
    def logger(cls) -> Logger:
        """Gets a logger for the particular Experiment subclass."""
        return get_logger(cls.__name__)

    @classmethod
    def result_cls(cls) -> type[R]:
        """Gets the Result subclass.
        Infers this from the parameter R of the Experiment[R] class from which this subclass should inherit."""
        for base in cls.__orig_bases__:
            if issubclass(base.__origin__, Experiment):
                return base.__args__[0]
        raise TypeError('Experiment subclass must inherit from Experiment[R] for some Result subclass R')

    @classmethod
    def extra_columns(cls) -> ColumnMap:
        """Gets additional columns to provide to the SQL table which are not included among the dataclass fields."""
        return {
            'id': Column('id', Integer, primary_key=True, autoincrement=True),
            'exp_id': Column('exp_id', String),
            'time': Column('time', DateTime)
        }

    @classmethod
    @cache
    def sql_cls(cls) -> type[SQLDataclass]:
        """Creates a custom subclass of SQLDataclass that has a sqlalchemy-backed SQL table."""
        flds = []
        for cl in [cls, cls.result_cls()]:
            for field in cl.get_fields():  # type: ignore
                has_default = (field.default is not MISSING) or (field.default_factory is not MISSING)
                # to preserve order, put a dummy default of None for any mandatory fields
                if not has_default:
                    field = copy(field)
                    field.default = None
                flds.append((field.name, field.type, field))
        dcl = make_dataclass(cls.__name__, flds, bases=(SQLDataclass,))
        return register(extra_cols = cls.extra_columns())(dcl)

    @abstractmethod
    def run(self) -> R | list[R]:
        """Runs the experiment, producing a Result or a list of Results."""


class ExperimentResults(TypedDict, Generic[R]):
    """Bundle of data returned by an experiment run."""
    experiment_cls: type[Experiment[R]]
    experiment_data: JSONDict
    time: datetime
    results: list[JSONDict]


def run_experiment(experiment: Experiment[R], errors: str = 'raise', verbosity: int = 0, debug: bool = False) -> Optional[ExperimentResults[R]]:  # type: ignore
    """Runs a single experiment."""
    if verbosity >= 2:
        LOGGER.info(str(experiment))
    experiment_data = experiment.to_dict()
    try:
        result = experiment.run()
        if isinstance(result, list):
            results = [res.to_dict() for res in result]
        else:  # single result
            results = [result.to_dict()]
        exp_results: ExperimentResults[R] = {
            'experiment_cls': experiment.__class__,
            'time': datetime.now(),
            'experiment_data': experiment_data,
            'results': results,
        }
        return exp_results
    except Exception as e:
        if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)):
            raise e
        if errors == 'raise':
            raise e
        if errors == 'warn':
            logger = experiment.logger()
            logger.error(f'{type(e).__name__}:\n\t{experiment_data}\n\tERROR: {e}')
            return None


@dataclass
class ExperimentRunner(Generic[R]):
    """Main driver for running experiments."""
    params: dict[type[Experiment[R]], Params]  # mapping from experiment class to parameters
    engine: Engine  # SQL engine
    verbosity: int = 0  # verbosity level
    errors: str = 'warn'  # how to handle errors (ignore, warn, raise)
    num_threads: int = 1  # number of threads to use
    chunk_size: int = 1  # number of experiments per chunk
    shuffle: bool = False  # shuffle the experiments
    no_rerun: bool = False  # do not re-run the same experiment if already in the database
    debug: bool = False  # drop into debugger if an error occurs (single-threaded only)

    def __post_init__(self) -> None:
        # ensure params are wrapped in the Params class
        self.params = {cls : Params(params) for (cls, params) in self.params.items()}

    def __iter__(self) -> Iterator[Experiment[R]]:
        for (cls, params) in self.params.items():
            yield from (cls.from_dict(d) for d in params)

    def result_classes(self) -> dict[type[Experiment[R]], type[SQLDataclass]]:
        """Gets a mapping from Experiment classes to SQLDataclasses storing both parameters and results."""
        return {cls: cls.sql_cls() for cls in self.params}

    def create_tables(self) -> None:
        """Creates SQL tables for every experiment."""
        # create all the tables
        for cls in self.params:
            cls.sql_cls()
        DEFAULT_REGISTRY.metadata.create_all(self.engine)

    def make_session(self) -> Session:
        """Creates a new SQLAlchemy session."""
        return sessionmaker(bind = self.engine)()

    def run(self) -> None:
        """Runs all experiments."""
        self.create_tables()
        session = self.make_session()
        experiments = list(self)
        LOGGER.info(f'Parameters specify {len(experiments):,d} experiments.')
        if self.no_rerun:
            LOGGER.info('Filtering out already-run experiments.')
            # filter out experiments already in the database
            def is_new_experiment(experiment: Experiment[R]) -> bool:
                sql_cls = experiment.sql_cls()
                flt = reduce(and_, (getattr(sql_cls, field) == getattr(experiment, field) for field in experiment.__dataclass_fields__))
                rows = session.query(sql_cls).filter(flt)
                try:
                    _ = next(iter(rows))
                    return False
                except StopIteration:
                    return True
            experiments = list(filter(is_new_experiment, experiments))
        if self.shuffle:
            random.shuffle(experiments)
        num_experiments = len(experiments)
        LOGGER.info(f'Running {num_experiments:,d} experiments with {self.num_threads} thread(s)...')
        pool = mp.Pool(self.num_threads)
        if self.num_threads == 1:
            mapper: Any = map
            debug = self.debug
        else:
            mapper = partial(pool.imap_unordered, chunksize=self.chunk_size)
            debug = False
        func = partial(run_experiment, errors=self.errors, verbosity=self.verbosity, debug=debug)
        all_results = mapper(func, experiments)
        exp_id = datetime.now().strftime('%Y%m%d%H%M%S')
        result_classes = self.result_classes()
        for results in tqdm(all_results, total = num_experiments):
            if (results is not None):
                experiment_cls = results['experiment_cls']
                result_cls = result_classes[experiment_cls]
                time = results['time']
                experiment_data = results['experiment_data']
                for result in results['results']:
                    # combine experiment and result data into a single entry
                    result_dict = {**experiment_data, **result}
                    res = result_cls.from_dict(result_dict)  # automatically infers subtype
                    res.exp_id = exp_id
                    res.time = time
                    session.add(res)
                    session.commit()
        LOGGER.info('\033[1m' + 'DONE!')
