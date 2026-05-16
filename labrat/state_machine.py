from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from fancy_dataclass import SQLDataclass

from labrat.experiment import Experiment, Result


# state type
S = TypeVar('S')  # state
# input type
I = TypeVar('I')  # noqa: E741
R = TypeVar('R', bound = Result)  # result
T = TypeVar('T')


@dataclass
class StateMachine(ABC, SQLDataclass, Generic[S, I]):
    """Abstract class representing a state machine."""

    @abstractproperty
    def start_state(self) -> S:
        """Gets the start state of the machine."""

    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns True if the given state is a final state."""

    @abstractmethod
    def transition(self, state: S, input: I) -> S:  # noqa: A002
        """Transitions from one state to another, given input."""


@dataclass
class StateMachineExperiment(Experiment[R], Generic[S, I, R]):
    state_machine: StateMachine[S, I]

    @abstractmethod
    def generate_inputs(self) -> Iterator[I]:
        """Generates a sequence of inputs to the state machine."""

    @abstractmethod
    def get_result(self, state: S) -> R:
        """Gets the result for the given state."""

    def run(self) -> R:
        state = self.state_machine.start_state
        for val in self.generate_inputs():
            if self.state_machine.is_final(state):
                break
            state = self.state_machine.transition(state, val)
        # finish looping if a final state is reached, or input sequence is exhausted
        return self.get_result(state)
