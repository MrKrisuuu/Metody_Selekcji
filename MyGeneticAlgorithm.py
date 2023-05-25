from typing import List, TypeVar
from jmetal.algorithm.singleobjective import GeneticAlgorithm  # Algorytmy
from jmetal.operator import SBXCrossover  # Cross
from jmetal.operator import PolynomialMutation  # Mutacja
from jmetal.util.termination_criterion import StoppingByEvaluations  # Warunek końca

import time
import copy
import os
import shutil

from selections import (
    MyNeuralNetworkSelection
)

from statistics import stdev

S = TypeVar("S")


def calc_stdev(solutions):
    tmps = [[] for _ in range(len(solutions[0].variables))]
    for solution in solutions:
        for i, var in enumerate(solution.variables):
            tmps[i].append(var)
    tmp_stdev = 0
    for vars in tmps:
        tmp_stdev += stdev(vars)
    return tmp_stdev


def save_stdev(stdevs, problem, selection, path="./stdevs"):
    with open(f"{path}/{problem.get_name()}/{selection.get_name()}.txt", "a") as f:
        for stdev in stdevs:
            f.write(str(stdev))
            f.write(';')
        f.write('\n')


class MyGeneticAlgorithm(GeneticAlgorithm):
    def __init__(
        self, steps, problem, population_size, offspring_population_size, selection, std_count=0
    ):
        super(MyGeneticAlgorithm, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            selection=selection,
            termination_criterion=StoppingByEvaluations(
                max_evaluations=population_size + steps * offspring_population_size
            ),
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables, distribution_index=20
            ),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
        )
        self.std_count = std_count

    def run(self, initial_solutions=None):
        """Execute the algorithm."""
        self.start_computing_time = time.time()

        if isinstance(self.selection_operator, MyNeuralNetworkSelection):
            self.selection_operator.train_model(self.problem)

        if initial_solutions:
            self.solutions = copy.deepcopy(initial_solutions)
        else:
            self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        solutions = []
        stdevs = []
        while not self.stopping_condition_is_met():
            solutions.append(self.solutions[0])
            if len(stdevs) < self.std_count-1:
                stdevs.append(calc_stdev(self.solutions))
            self.step()
            self.update_progress()
        if self.std_count > 0:
            stdevs.append(calc_stdev(self.solutions))
            save_stdev(stdevs, self.problem, self.selection_operator)
        self.total_computing_time = time.time() - self.start_computing_time
        return solutions

    def selection(self, population: List[S]):
        mating_population = []

        while len(mating_population) < self.mating_pool_size:
            if (
                "flag_list" in self.selection_operator.__dict__.keys()
                and self.selection_operator.flag_list
            ):
                solution = self.selection_operator.execute(
                    population, self.mating_pool_size
                )
            else:
                solution = self.selection_operator.execute(population)

            if isinstance(solution, list):
                mating_population = mating_population + solution
            else:
                mating_population.append(solution)

        return mating_population[:self.mating_pool_size]
