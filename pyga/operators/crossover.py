"""The :mod:`~pyga.operators.crossover` contains crossover operators"""
import numpy as np
import pyga.encoding as en


def single_point(pc, parents):
    """Use :func:`point <pyga.operators.crossover.single_point>` to do a single point crossover between parents. The
    crossover point is selected randomly.

    Args:
        pc (float): The probability with which crossover occurs.
        parents (tuple(pyga.encoding.BinaryEncoding)): The parents that produce offsprings.

    Returns:
        tuple(pyga.encoding.BinaryEncoding): Two offspring.

    """
    if np.random.rand() < pc:
        # crossover occurs
        parent1 = parents[0]
        parent2 = parents[1]
        crossover_point = np.random.randint(1, len(parents))
        offspring1 = en.BinaryEncoding(parent1[:crossover_point] + parent2[crossover_point:])
        offspring2 = en.BinaryEncoding(parent2[:crossover_point] + parent1[crossover_point:])
        return offspring1, offspring2
    return parents
