from jmetal.operator.selection import *  # Selekcja

import numpy as np
from scipy.stats import cauchy

from .utils import save


class MyNormalFadingSelection(Selection[List[S], S]):
    def __init__(self, width_constant=0.5, fade_constant=0.999):
        super(MyNormalFadingSelection, self).__init__()
        self.flag_list = True
        self.fade = 1
        self.width_constant = width_constant
        self.fade_constant = fade_constant

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
                np.random.normal(-sol.objectives[0], width * self.width_constant * self.fade) for sol in front
            ]
            self.fade *= self.fade_constant
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return f"MyNormalFadingSelection({self.width_constant}, {self.fade_constant})"


class MyCauchyFadingSelection(Selection[List[S], S]):
    def __init__(self, width_constant=0.05, fade_constant=0.999):
        super(MyCauchyFadingSelection, self).__init__()
        self.flag_list = True
        self.fade = 1
        self.width_constant = width_constant
        self.fade_constant = fade_constant

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
            deltas = cauchy.rvs(0, width * self.width_constant * self.fade, len(front))
            sums = [-sol.objectives[0] + deltas[i] for i, sol in enumerate(front)]
            self.fade *= self.fade_constant
            # save(sols, sums, number, self.get_name())
            my_result = list(zip(front, sums))
            my_result.sort(key=lambda ind: ind[1], reverse=True)
            result = [sol for sol, _ in my_result]
        return result[:number]

    def get_name(self) -> str:
        return f"MyCauchyFadingSelection({self.width_constant}, {self.fade_constant})"