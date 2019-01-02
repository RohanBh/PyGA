"""The :mod:`~pyga.operators.mutation` contains different mutation strategies."""

import random

from scipy import stats


def random_bit(offspring, p, inversion=False):
    """Use :func:`~pyga.operators.mutation.random_bit` to apply bit mutation operator to an offspring.

    Args:
        offspring (pyga.encoding.BinaryEncoding): An offspring represented by a bit string.
        p (float): The mutation probability. Each bit of the offspring is susceptible to mutation with probability pm.
        inversion (bool): Whether inversion is being used or not. Default is False.

    Returns:
         The mutated offspring.
    """
    mutation_index = stats.bernoulli.rvs(p=p, size=len(offspring)) != 0
    if inversion:
        offspring.ivalue[mutation_index, 1] ^= True
    else:
        offspring.value[mutation_index] ^= True
    return offspring


def inversion(offspring, p):
    """Applies inversion operator to offspring.

    Args:
        offspring (pyga.encoding.Encoding): An offspring to be inverted
        p (float): Probability with which inversion occurs.

    Returns:
        pyga.encoding.Encoding: An (possibly) inverted offspring.
    """
    if random.random() <= p:
        a = random.randrange(0, len(offspring))
        b = random.randrange(0, len(offspring))
        a, b = (a, b) if a < b else (b, a)
        offspring.ivalue[a:b] = offspring.ivalue[b - 1:a - 1:-1]
    return offspring
