from abc import ABC, abstractmethod
import bdb
from dataclasses import dataclass
from datetime import datetime
import inspect
from logging import Logger
from typing import Annotated, Any, ClassVar, Generic, Literal, Optional, TypeAlias, TypeVar, cast, get_args, get_origin

from fancy_dataclass import JSONDataclass
from typing_extensions import Doc

from labrat import LOGGER, get_logger


T = TypeVar('T')

# how to handle errors in experiments
ErrorMode: TypeAlias = Literal['ignore', 'warn', 'raise']


class Result(JSONDataclass, store_type='off'):
    """A class for experimental results, where the fields can be mapped to a SQL table."""


R = TypeVar('R', bound=Result)


@dataclass
class Experiment(JSONDataclass, Generic[R], ABC, store_type='off', allow_extra_fields=False):
    """A type representing some kind of computational experiment.

    The type itself corresponds to a SQL table.

    An instance of an Experiment represents an experimental trial with one particular choice of parameters.
    The result will be stored in the type's SQL table."""
    result_type: ClassVar[type[R]]  # stores the result type so it can be accessed at runtime

    @staticmethod
    def _get_result_type_arg(cls: type['Experiment[Any]']) -> Optional[type[Result]]:
        for base in cls.__orig_bases__:  # type: ignore[attr-defined]
            if (origin := get_origin(base)) and issubclass(origin, Experiment):
                for arg in get_args(base):
                    if isinstance(arg, type) and issubclass(arg, Result):
                        return arg
        return None

    def __init_subclass__(cls, *args: type, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)  # must forward kwargs
        if not inspect.isabstract(cls):
            if hasattr(cls, 'result_type'):
                result_type: Optional[type[R]] = cls.result_type
            else:
                # infer the result type from the first generic parameter whose type is a subclass of Result
                result_type = cast(type[R], Experiment._get_result_type_arg(cls))
            if result_type is None:
                raise TypeError(f'{cls.__name__} must define a result_type ClassVar')
            if not (isinstance(result_type, type) and issubclass(result_type, Result)):
                raise TypeError(f'result_type for {cls.__name__} must be a subclass of {Result.__qualname__}')
            cls.result_type = result_type

    @classmethod
    def logger(cls) -> Logger:
        """Gets a logger for this particular Experiment subclass."""
        return get_logger(cls.__name__)

    @abstractmethod
    def run(self) -> R:
        """Runs the experiment, returning a result of type R."""


@dataclass(frozen=True)
class ExperimentResult(Generic[R]):
    """Bundle of data returned when running an experiment."""
    session_id: Annotated[
        Optional[str],
        Doc('session ID string for an experiment run'),
    ]
    experiment: Annotated[
        Experiment[R],
        Doc('experiment that was run'),
    ]
    start_time: Annotated[
        datetime,
        Doc('time the experiment was started'),
    ]
    end_time: Annotated[
        datetime,
        Doc('time the experiment was completed'),
    ]
    result: Annotated[
        R | Exception,
        Doc('result of the experiment (or an error)'),
    ]


def run_experiment(
    experiment: Experiment[R],
    error_mode: ErrorMode = 'raise',
    verbosity: int = 0,
    debug: bool = False,
) -> ExperimentResult[R]:
    """Runs an Experiment, returning an ExperimentResult object which wraps a raw result."""
    if verbosity >= 2:
        LOGGER.info(str(experiment))
    try:
        start_time = datetime.now()
        result: R | Exception = experiment.run()
    except Exception as e:
        if debug:
            import pdb  # noqa: T100
            pdb.post_mortem()
        if isinstance(e, (KeyboardInterrupt, bdb.BdbQuit)) or (error_mode == 'raise'):
            raise
        if error_mode == 'warn':
            logger = experiment.logger()
            logger.error(f'{type(e).__name__}:\n\t{experiment}\n\tERROR: {e}')
        # store the exception object itself in the result
        result = e
    return ExperimentResult(
        session_id=None,
        experiment=experiment,
        start_time=start_time,
        end_time=datetime.now(),
        result=result,
    )
