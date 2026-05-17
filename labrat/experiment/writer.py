from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import suppress
from copy import copy
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass, make_dataclass
from functools import cache, cached_property, reduce
from typing import Any, TypeVar

from fancy_dataclass.sql import DEFAULT_REGISTRY, SQLDataclass, register
import sqlalchemy
from sqlalchemy import Column, Engine, Integer, create_engine
from sqlalchemy.orm import Session, sessionmaker

from labrat.experiment._base import Experiment, ExperimentResult, Result


R = TypeVar('R', bound=Result)


def _generate_result_entry_type_fields(experiment_type: type[Experiment[R]]) -> Iterator[Field[Any]]:
    """Given an Experiment subclass, iterates over dataclass fields for the type itself and then those of its
    result type."""
    for fld in fields(ExperimentResult):
        match fld.name:
            case 'experiment':
                assert is_dataclass(experiment_type)
                yield from fields(experiment_type)
            case 'result':
                result_type = experiment_type.result_type()
                assert is_dataclass(result_type)
                yield from fields(result_type)
            case _:
                yield fld

@cache
def get_result_entry_type(experiment_type: type[Experiment[R]]) -> type[SQLDataclass]:
    """Given an Experiment subclass, creates a new SQLDataclass subclass corresponding to a SQL table entry combining
    both the experiment parameters and the result.
    The name of the created class will be the same as the name of the Experiment subclass."""
    flds = []
    field_names = set()
    for fld in _generate_result_entry_type_fields(experiment_type):
        if fld.name in field_names:
            raise ValueError(f'duplicate field name {fld.name!r}')
        field_names.add(fld.name)
        has_default = (fld.default is not MISSING) or (fld.default_factory is not MISSING)
        # to preserve order, put a dummy default of None for any mandatory fields
        if not has_default:
            fld = copy(fld)
            fld.default = None
        flds.append((fld.name, fld.type, fld))
    dcls = make_dataclass(experiment_type.__name__, flds, bases=(SQLDataclass,))
    if 'id' in field_names:
        extra_cols = {}
    else:
        # make 'id' a primary key auto-incrementing column
        extra_cols = {'id': Column('id', Integer, primary_key=True, autoincrement=True)}
    return register(extra_cols=extra_cols)(dcls)


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
            _ = get_result_entry_type(experiment_type)
        DEFAULT_REGISTRY.metadata.create_all(self.engine)
        # create the Session object for performing database actions
        _ = self.session

    def experiment_was_written(self, experiment: Experiment[Any]) -> bool:
        # check if any result for this experiment is already in the database
        result_entry_type = get_result_entry_type(type(experiment))
        flt = reduce(
            sqlalchemy.and_,
            (getattr(result_entry_type, fld.name) == getattr(experiment, fld.name) for fld in fields(experiment)),
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
        result_entry_type = get_result_entry_type(experiment_type)
        # combine experiment and result data into a single table entry
        entry = result_entry_type(
            **result.experiment.to_dict(),
            **result.result.to_dict(),
            **{
                'session_id': session_id,
                'start_time': result.start_time,
                'end_time': result.end_time,
            },
        )
        # write the result to the database
        self.session.add(entry)
        self.session.commit()


def sql_writer(engine: str | Engine) -> SQLExperimentResultWriter:
    """Convenience function for constructing a SQLExperimentResultWriter."""
    return SQLExperimentResultWriter(engine)
