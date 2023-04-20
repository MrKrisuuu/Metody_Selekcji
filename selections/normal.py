from jmetal.operator.selection import *  # Selekcja

from typing import List, TypeVar
from .utils import save


import numpy as np


S = TypeVar("S")


class MyNormalSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyNormalSelection, self).__init__()
        self.flag_list = True

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            sols = [sol.objectives[0] for sol in front]
            width = max(sols) - min(sols)
            sums = [np.random.normal(-sol.objectives[0], width / 4) for sol in front]
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyNormalSelection"


class MyNormalFadingSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyNormalFadingSelection, self).__init__()
        self.flag_list = True
        self.fade = 1

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            sols = [sol.objectives[0] for sol in front]
            width = max(sols) - min(sols)
            sums = [
                np.random.normal(-sol.objectives[0], self.fade * width) for sol in front
            ]
            self.fade *= 0.99
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyNormalFadingSelection"
