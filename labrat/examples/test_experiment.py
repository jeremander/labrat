from dataclasses import dataclass
import time

from sqlalchemy import create_engine

from labrat.experiment import Experiment, ExperimentRunner, Result
from labrat.params import Params


@dataclass
class MyResult(Result):
    d: int
    e: bool

@dataclass
class Experiment1(Experiment[MyResult]):
    a: int
    b: str
    c: float
    def run(self) -> MyResult:
        time.sleep(0.1)
        if self.a == 13:
            raise ValueError('13 is unlucky!')
        return MyResult(self.a, self.b == 'b')

@dataclass
class Experiment2(Experiment[MyResult]):
    x: str
    y: int
    def run(self) -> MyResult:
        time.sleep(0.1)
        return MyResult(self.y, self.x == 'abc')


if __name__ == '__main__':

    engine = create_engine('sqlite:///test.sqlite')
    params1 = Params({'a' : list(range(20)), 'b' : ['a', 'b', 'c'], 'c': [1.0, 10.0]})
    params2 = Params({'x' : ['abc', 'def'], 'y' : [1, 2]})
    params = {Experiment1 : params1, Experiment2 : params2}

    runner = ExperimentRunner(params, engine, num_threads=4)
    runner.run()
