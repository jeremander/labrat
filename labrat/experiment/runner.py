from dataclasses import dataclass
from datetime import datetime
from functools import partial
import random
from typing import Annotated, Any
import warnings

from typing_extensions import Doc

from labrat import LOGGER
from labrat.experiment._base import ErrorMode, Experiment, run_experiment
from labrat.experiment.writer import ExperimentResultWriter
from labrat.params import Params
from labrat.utils import parallel_map


@dataclass(frozen=True)
class ExperimentRunner:
    """Main driver for running experiments."""
    params: Annotated[
        dict[type[Experiment[Any]], Params],
        Doc('mapping from experiment class to parameters'),
    ]
    result_writer: Annotated[
        ExperimentResultWriter,
        Doc('object responsible for writing the results'),
    ]
    verbosity: Annotated[
        int,
        Doc('verbosity level'),
    ] = 0
    error_mode: Annotated[
        ErrorMode,
        Doc('how to handle errors (ignore, warn, raise)'),
    ] = 'warn'
    num_threads: Annotated[
        int,
        Doc('number of threads to use'),
    ] = 1
    chunk_size: Annotated[
        int,
        Doc('number of experiments per chunk, when parallelizing'),
    ] = 1
    shuffle: Annotated[
        bool,
        Doc('randomly shuffle the experiments'),
    ] = False
    no_rerun: Annotated[
        bool,
        Doc('do not re-run the same experiment if its result was already written'),
    ] = False
    debug: Annotated[
        bool,
        Doc('drop into debugger if an error occurs (single-threaded only)'),
    ] = False

    def __post_init__(self) -> None:
        if self.debug and (self.num_threads > 1):
            warnings.warn('debug = True will have no effect when num_threads > 1', UserWarning, stacklevel=1)

    def get_experiments(self) -> list[Experiment[Any]]:
        return [cls.from_dict(d) for (cls, params) in self.params.items() for d in params]

    def _get_experiments(self) -> list[Experiment[Any]]:
        experiments = self.get_experiments()
        num_experiments = len(experiments)
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
        debug = self.debug and (self.num_threads <= 1)
        func = partial(run_experiment, error_mode=self.error_mode, verbosity=self.verbosity, debug=debug)
        # if shuffling experiments, no need to return the results in the original order
        ordered = not self.shuffle
        all_results = parallel_map(func, experiments, num_threads=self.num_threads, ordered=ordered, progress=True)
        # TODO: use milliseconds? Use a hash of the experiment data instead?
        # create ID for the entire session of experiment runs
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        LOGGER.info('Processing results...')
        for result in all_results:
            # TODO: handle errors
            if result is not None:
                self.result_writer.write_experiment_result(session_id, result)
        LOGGER.info('\033[1m' + 'DONE!')
