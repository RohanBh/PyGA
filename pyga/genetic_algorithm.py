"""The :mod:`~pyga.genetic_algorithm` provides functions for different implementations of genetic algorithms"""
from pyga.operators import selection, crossover, mutation


def simple_ga(encodings, generations=50, pc=0.7, pm=0.001, pi=None):
    """A simple Holland's original genetic algorithm involving fitness proportionate selection, single point crossover
     and random mutation over a population whose individual is represented by a binary string (bitstring).

    Args:
        encodings (pyga.encoding.BinaryEncoding): A numpy array of encodings. It is None by default in which case, it is
        initialised randomly.
        generations (int): Number of generations over which the GA should run. Default is 50.
        pc (float): The crossover probability.
        pm (float): The mutation probability.
        pi (float): The inversion probability. Default is None, in which case, inversion is not used.
    """
    use_inversion = pi is None
    for i in range(generations):
        next_gen_encodings = []

        for parent1, parent2 in selection.roulette_wheel(encodings):
            offspring1, offspring2 = crossover.single_point((parent1, parent2), pc, use_inversion)
            offspring1 = mutation.random_bit(offspring1, pm, use_inversion)
            offspring2 = mutation.random_bit(offspring2, pm, use_inversion)
            if use_inversion:
                offspring1 = mutation.inversion(offspring1, pi)
                offspring2 = mutation.inversion(offspring2, pi)
            next_gen_encodings.append(offspring1)
            next_gen_encodings.append(offspring2)

        encodings = next_gen_encodings
    return encodings
