"""The :mod:`~pyga.operators.mutation` contains different mutation strategies."""
import numpy as np
from scipy import stats


def random_mutation(pm, offspring):
    """Use :func:`~pyga.operators.mutation.random_mutation` to apply mutation operator to an offspring.

    Args:
        pm (float): The mutation probability. Each bit of the offspring is susceptible to mutation with probability pm.
        offspring (pyga.encoding.BinaryEncoding): An offspring represented by a bit string.

    Returns:
         The mutated offspring.
    """
    mutation_index = stats.bernoulli.rvs(p=pm, size=len(offspring)) != 0
    mutated_value = (offspring.value[mutation_index] ^ True)
    offspring.value = mutated_value
    return offspring
