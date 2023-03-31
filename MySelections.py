from jmetal.core.operator import Selection
from jmetal.operator.selection import *  # Selekcja

from typing import List, TypeVar
from random import sample

import tensorflow as tf


S = TypeVar("S")


class MySelection(Selection[List[S], S]):
    def __init__(self):
        super(MySelection, self).__init__()

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            # Sampling without replacement
            i, j = sample(range(0, len(front)), 2)
            solution1 = front[i]
            solution2 = front[j]

            flag = solution1.objectives[0] < solution2.objectives[0]

            if flag:
                result = solution1
            else:
                result = solution2

        return result

    def get_name(self) -> str:
        return "MySelection"


class MyNeuralNetworkSelection(Selection[List[S], S]):
    def __init__(self, model: tf.keras.models.Sequential):
        super(MyNeuralNetworkSelection, self).__init__()
        self.model = model

    def execute(self, front: List[S]) -> S:
        if front is None:
            raise Exception("The front is null")
        elif len(front) == 0:
            raise Exception("The front is empty")

        if len(front) == 1:
            result = front[0]
        else:
            a, b = np.random.choice(front, 2)
            print(np.array([a.variables + b.variables]).shape)
            result = self.model([a.variables + b.variables])

            print(result)

        return result

    def get_name(self) -> str:
        return "MyNeuralNetworkSelection"
