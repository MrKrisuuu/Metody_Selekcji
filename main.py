from MySelections import MySelection
from MyFunctions import RastriginFunction

from jmetal.algorithm.singleobjective import GeneticAlgorithm  # Algorytmy
from jmetal.operator.selection import *  # Selekcja
from jmetal.operator import SBXCrossover  # Cross
from jmetal.operator import PolynomialMutation  # Mutacja
from jmetal.util.termination_criterion import StoppingByEvaluations  # Warunek końca
from jmetal.config import store
from jmetal.problem.singleobjective.unconstrained import Rastrigin

import matplotlib.pyplot as plt
import time


class MyGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, steps, problem, population_size, offspring_population_size, selection):
        super(MyGeneticAlgorithm, self).__init__(problem=problem,
                                                 population_size=population_size,
                                                 offspring_population_size=offspring_population_size,
                                                 selection=selection,
                                                 termination_criterion=StoppingByEvaluations(
                                                     max_evaluations=population_size + steps * offspring_population_size),
                                                 mutation=PolynomialMutation(
                                                     probability=1.0 / problem.number_of_variables,
                                                     distribution_index=20),
                                                 crossover=SBXCrossover(probability=1.0, distribution_index=20))

    def run(self, initial_solutions=None):
        """ Execute the algorithm. """
        self.start_computing_time = time.time()

        if initial_solutions:
            self.solutions = initial_solutions
        else:
            self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        solutions = []
        while not self.stopping_condition_is_met():
            solutions.append(self.solutions[0])
            self.step()
            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time
        return solutions


def create_initial_solutions(problem, population_size) -> List[S]:
    return [store.default_generator.new(problem)
            for _ in range(population_size)]


def run_selections(problems, selections):
    population_size = 100
    offspring_population_size = 100
    steps = 1000
    for problem in problems:
        initial_solutions = create_initial_solutions(problem, population_size)
        for selection in selections:
            algorithm = MyGeneticAlgorithm(
                steps=steps,
                problem=problem,
                population_size=population_size,
                offspring_population_size=offspring_population_size,
                selection=selection
            )
            solutions = algorithm.run(initial_solutions)
            best_solution = algorithm.get_result()
            solutions.append(best_solution)
            # for i, solution in enumerate(solutions):
            #     print(f"{i}:", solution)
            results = [solution.objectives[0] for solution in solutions]
            print(f"{selection.get_name()} for {problem.get_name()}: {best_solution.objectives[0]} in {algorithm.total_computing_time} seconds")
            plt.plot(range(steps + 1), results, label=selection.get_name())
        plt.yscale("log")
        plt.legend()
        plt.show()


problems = [RastriginFunction(50)]
selections = [BestSolutionSelection(),
              BinaryTournamentSelection(),
              RandomSolutionSelection()]
run_selections(problems, selections)
