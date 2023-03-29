from jmetal.algorithm.singleobjective import GeneticAlgorithm  # Algorytmy
from jmetal.operator import SBXCrossover  # Cross
from jmetal.operator import PolynomialMutation  # Mutacja
from jmetal.util.termination_criterion import StoppingByEvaluations  # Warunek końca

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