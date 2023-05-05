from jmetal.operator.selection import *  # Selekcja

import numpy as np
from scipy.stats import cauchy

from .utils import save
from itertools import combinations


class MyBestSolutionSelection(Selection[List[S], S]):
    def __init__(self):
        super(MyBestSolutionSelection, self).__init__()
        self.flag_list = True

    def execute(self, front: List[S], number=None) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")
        return front[0]

    def get_name(self) -> str:
        return "MyBestSolutionSelection"
