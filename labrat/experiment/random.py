from abc import abstractmethod
from dataclasses import dataclass
import hashlib
from typing import Generic, Optional, TypeVar

from labrat.experiment._base import Experiment, Result


R = TypeVar('R', bound=Result)
S = TypeVar('S')


MAX_SEED = 2 ** 32


def get_trial_seed(base_seed: int, trial: int) -> int:
    """Given a base seed and trial index, creates a new seed specific to the trial.
    Uses a SHA-256 hash to "mix" the base seed and trial together in an unpredictable way."""
    h = hashlib.sha256(f'{base_seed}:{trial}'.encode()).digest()
    return int.from_bytes(h[:8], 'little') % MAX_SEED


@dataclass
class RandomExperiment(Experiment[R]):
    """An experiment with some randomness involved.

    Such experiments are typically run with multiple trials."""
    # index of a trial
    trial: int = 0
    # base seed for all trials of the experiment
    base_seed: Optional[int] = None

    def get_trial_seed(self) -> Optional[int]:
        """Gets a random seed specific to the trial, if a base_seed is set.
        Otherwise, returns None."""
        return None if (self.base_seed is None) else get_trial_seed(self.base_seed, self.trial)

    @abstractmethod
    def run_with_seed(self, seed: Optional[int]) -> R:
        """Given an optional random seed, runs the experiment."""

    def run(self) -> R:
        return self.run_with_seed(self.get_trial_seed())


@dataclass
class MonteCarloExperiment(Generic[S, R], RandomExperiment[R]):
    """A Monte Carlo experiment, which generates random samples and then evaluates them."""

    @abstractmethod
    def sample(self, seed: Optional[int]) -> S:
        """Given a seed, produces a random sample."""

    @abstractmethod
    def result_from_sample(self, sample: S) -> R:
        """Given a sample, gets a result."""

    def run_with_seed(self, seed: Optional[int]) -> R:
        return self.result_from_sample(self.sample(seed))
