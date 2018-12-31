"""The :mod:`~pyga.operators.mutation` contains different mutation strategies."""
from scipy import stats
import random


def random_bit(offspring, p):
    """Use :func:`~pyga.operators.mutation.random_bit` to apply bit mutation operator to an offspring.

    Args:
        offspring (pyga.encoding.BinaryEncoding): An offspring represented by a bit string.
        p (float): The mutation probability. Each bit of the offspring is susceptible to mutation with probability pm.

    Returns:
         The mutated offspring.
    """
    mutation_index = stats.bernoulli.rvs(p=p, size=len(offspring)) != 0
    mutated_value = (offspring.value[mutation_index] ^ True)
    offspring.value = mutated_value
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
        offspring.ivalue[a:b] = offspring.ivalue[b-1:a-1:-1]
    return offspring
