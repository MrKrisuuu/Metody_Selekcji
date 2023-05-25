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

from math import cos, pi, sin, sqrt


class Sphere(FloatProblem):
    def __init__(self, number_of_variables: int = 10, steps: int = 1000):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.steps = steps

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

        return solution

    def get_name(self) -> str:
        return f"Sphere({self.number_of_variables}, {self.steps})"


class Rastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 10, steps: int = 1000):
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

        self.steps = steps

    def evaluate(self, solution):
        total = 10 * len(solution.variables)
        for x in solution.variables:
            total += (x-250) * (x-250) - 10 * cos(2 * pi * (x-250))
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return f"Rastrigin({self.number_of_variables}, {self.steps})"


class Rosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 10, steps: int = 1000):
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

        self.steps = steps

    def evaluate(self, solution):
        total = 0
        for i in range(len(solution.variables)-1):
            total += 100*(solution.variables[i+1] - solution.variables[i]**2)**2 + (solution.variables[i]-1)**2
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return f"Rosenbrock({self.number_of_variables}, {self.steps})"


class Schwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 10, steps: int = 1000):
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

        self.steps = steps

    def evaluate(self, solution):
        total = 418.9829*len(solution.variables)
        for x in solution.variables:
            total -= x*sin(sqrt(abs(x)))
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return f"Schwefel({self.number_of_variables}, {self.steps})"


class Griewank(FloatProblem):
    def __init__(self, number_of_variables: int = 10, steps: int = 1000):
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

        self.steps = steps

    def evaluate(self, solution):
        total = 1
        for x in solution.variables:
            total += (x-250)**2/4000
        tmp = 1
        for i, x in enumerate(solution.variables):
            tmp *= cos((x-250)/sqrt(i+1))
        total -= tmp
        solution.objectives[0] = total
        return solution

    def get_name(self):
        return f"Griewank({self.number_of_variables}, {self.steps})"