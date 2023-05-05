from jmetal.operator.selection import *  # Selekcja

import numpy as np
from scipy.stats import cauchy

from .utils import save
from itertools import combinations


class MyNormalPairwiseComparisonSelection(Selection[List[S], S]):
    def __init__(self, width_constant=2):
        super(MyNormalPairwiseComparisonSelection, self).__init__()
        self.flag_list = True
        self.width_constant = width_constant

    def evaluate(self, sol1, sol2, width):
        return sol2.objectives[0] - sol1.objectives[0] + np.random.normal(0, width * self.width_constant)

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
        return f"MyNormalPairwiseComparisonSelection({self.width_constant})"


class MyCauchyPairwiseComparisonSelection(Selection[List[S], S]):
    def __init__(self, width_constant=0.2):
        super(MyCauchyPairwiseComparisonSelection, self).__init__()
        self.flag_list = True
        self.width_constant = width_constant

    def evaluate(self, sol1, sol2, delta):
        return sol2.objectives[0] - sol1.objectives[0] + delta

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
            n = len(front)
            deltas = cauchy.rvs(0, width * self.width_constant, int((n)*(n-1)/2))
            for i, tupl in enumerate(combinations(range(len(front)), 2)):
                x, y = tupl
                val = self.evaluate(front[x], front[y], deltas[i])
                matrix[x][y] = val
                matrix[y][x] = -val
            sums = [sum(row) for row in matrix]
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return f"MyCauchyPairwiseComparisonSelection({self.width_constant})"
