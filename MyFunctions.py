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
from jmetal.problem.singleobjective.unconstrained import Sphere

from math import cos, pi, sin, sqrt


class Rastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1  # Po ilu funkcjach minimalizujemy
        self.number_of_variables = number_of_variables  # Liczba zmiennych
        self.number_of_constraints = 0

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['f(x)']

        # Dolna granica zmiannych
        self.lower_bound = [-500 for _ in range(number_of_variables)]
        self.upper_bound = [
            500 for _ in range(number_of_variables)
        ]  # Górna granica zmiennych

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        total = 10 * len(solution.variables)
        for x in solution.variables:
            total += (x-250) * (x-250) - 10 * cos(2 * pi * (x-250))
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return "Rastrigin"


class Rosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Rosenbrock, self).__init__()
        self.number_of_objectives = 1  # Po ilu funkcjach minimalizujemy
        self.number_of_variables = number_of_variables  # Liczba zmiennych
        self.number_of_constraints = 0

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['f(x)']

        # Dolna granica zmiannych
        self.lower_bound = [-500 for _ in range(number_of_variables)]
        self.upper_bound = [
            500 for _ in range(number_of_variables)
        ]  # Górna granica zmiennych

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        total = 0
        for i in range(len(solution.variables)-1):
            total += 100*(solution.variables[i+1] - solution.variables[i]**2)**2 + (solution.variables[i]-1)**2
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return "Rosenbrock"


class Schwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Schwefel, self).__init__()
        self.number_of_objectives = 1  # Po ilu funkcjach minimalizujemy
        self.number_of_variables = number_of_variables  # Liczba zmiennych
        self.number_of_constraints = 0

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['f(x)']

        # Dolna granica zmiannych
        self.lower_bound = [-500 for _ in range(number_of_variables)]
        self.upper_bound = [
            500 for _ in range(number_of_variables)
        ]  # Górna granica zmiennych

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        total = 418.9829*len(solution.variables)
        for x in solution.variables:
            total -= x*sin(sqrt(abs(x)))
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return "Schwefel"


class Griewank(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Griewank, self).__init__()
        self.number_of_objectives = 1  # Po ilu funkcjach minimalizujemy
        self.number_of_variables = number_of_variables  # Liczba zmiennych
        self.number_of_constraints = 0

        # self.obj_directions = [self.MAXIMIZE]
        # self.obj_labels = ['f(x)']

        # Dolna granica zmiannych
        self.lower_bound = [-500 for _ in range(number_of_variables)]
        self.upper_bound = [
            500 for _ in range(number_of_variables)
        ]  # Górna granica zmiennych

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution):
        total = 0
        for x in solution.variables:
            total += x**2/4000
        tmp = 1
        for i, x in enumerate(solution.variables):
            tmp *= cos((x-250)/sqrt(i+1))
        total -= tmp
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return "Griewank"