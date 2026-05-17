from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, fields
from functools import cached_property, reduce
from typing import Any, TypeVar

from fancy_dataclass.sql import DEFAULT_REGISTRY
import sqlalchemy
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from labrat.experiment._base import Experiment, ExperimentResult, Result


R = TypeVar('R', bound=Result)


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
            sqlalchemy.and_,
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
