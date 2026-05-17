from dataclasses import dataclass
import time

from labrat.experiment import Experiment, ExperimentRunner, Result, sql_writer
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

    engine = 'sqlite:///test.sqlite'
    params1 = Params.grid({'a' : list(range(20)), 'b' : ['a', 'b', 'c'], 'c': [1.0, 10.0]})
    params2 = Params.grid({'x' : ['abc', 'def'], 'y' : [1, 2]})
    params: dict[type[Experiment[MyResult]], Params] = {Experiment1: params1, Experiment2: params2}

    result_writer = sql_writer('sqlite:///test.sqlite')
    num_threads = 1
    runner = ExperimentRunner(params, result_writer, num_threads=num_threads)
    runner.run()
