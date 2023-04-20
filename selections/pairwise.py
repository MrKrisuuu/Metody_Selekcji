from jmetal.operator.selection import *  # Selekcja

from typing import List
from .utils import save

from itertools import combinations

import numpy as np


class MyNormalPairwiseComparisonSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyNormalPairwiseComparisonSelection, self).__init__()
        self.flag_list = True

    def evaluate(self, sol1, sol2, width):
        val1 = np.random.normal(sol1.objectives[0], width)
        val2 = np.random.normal(sol2.objectives[0], width)
        return val2 - val1

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            matrix = [[0 for _ in range(len(front))] for _ in range(len(front))]
            sols = [sol.objectives[0] for sol in front]
            width = max(sols) - min(sols)
            for x, y in combinations(range(len(front)), 2):
                val = self.evaluate(front[x], front[y], width)
                matrix[x][y] = val
                matrix[y][x] = -val
            sums = [sum(row) for row in matrix]
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyNormalPairwiseComparisonSelection"


class MyOptimizedNormalPairwiseComparisonSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyOptimizedNormalPairwiseComparisonSelection, self).__init__()
        self.flag_list = True

    def evaluate(self, sol1, sol2, width):
        return sol1.objectives[0] - sol2.objectives[0] + np.random.normal(0, 2 * width)

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            matrix = [[0 for _ in range(len(front))] for _ in range(len(front))]
            sols = [sol.objectives[0] for sol in front]
            width = max(sols) - min(sols)
            for x, y in combinations(range(len(front)), 2):
                val = self.evaluate(front[x], front[y], width)
                matrix[x][y] = val
                matrix[y][x] = -val
            sums = [sum(row) for row in matrix]
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return "MyOptimizedNormalPairwiseComparisonSelection"
