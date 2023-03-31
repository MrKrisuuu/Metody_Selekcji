from jmetal.core.problem import (
    BinaryProblem,
    FloatProblem,
    IntegerProblem,
    PermutationProblem,
)  # Typ problemu
from jmetal.core.solution import (
    BinarySolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
)  # Typ rozwiązania
from jmetal.problem.singleobjective.unconstrained import Rastrigin, Sphere


from math import cos, pi


class RastriginFunction(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(RastriginFunction, self).__init__()
        self.number_of_objectives = 1  # Po ilu funkcjach minimalizujemy
        self.number_of_variables = number_of_variables  # Liczba zmiennych
        self.number_of_constraints = 0

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['f(x)']

        # Dolna granica zmiannych
        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [
            5.12 for _ in range(number_of_variables)
        ]  # Górna granica zmiennych

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        total = 10 * len(solution.variables)
        for x in solution.variables:
            total += x * x - 10 * cos(2 * pi * x)
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return "RastriginFunction"
