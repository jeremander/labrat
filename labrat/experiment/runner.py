from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, partial
import random
from typing import Any

from labrat import LOGGER
from labrat.experiment._base import ErrorMode, Experiment, ResultEntry
from labrat.experiment.writer import ExperimentResultWriter
from labrat.params import Params
from labrat.utils import parallel_map


@dataclass(frozen=True)
class ExperimentRunner:
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

    def __post_init__(self) -> None:
        if self.debug and (self.num_threads > 1):
            raise ValueError('cannot set debug = True when num_threads > 1')

    def get_experiments(self) -> list[Experiment[Any]]:
        return [cls.from_dict(d) for (cls, params) in self.params.items() for d in params]

    @cached_property
    def result_entry_types(self) -> dict[type[Experiment[Any]], type[ResultEntry]]:
        """Gets a mapping from Experiment classes to ResultEntry classes which store both parameters and results."""
        return {cls: cls.result_entry_type() for cls in self.params}

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
        func = partial(Experiment.get_result, error_mode=self.error_mode, verbosity=self.verbosity, debug=self.debug)
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
