"""The :mod:`~pyga.operators.selection` contains different selection strategies."""
import numpy as np


def roulette_wheel(encodings, size=None):
    """A roulette wheel selection function where each encoding is selected w.r.t its fitness.

    Args:
        encodings (pyga.encoding.Encoding): Possible parents to select from.
        size (Optional[int]): The number of offsprings in next generation. Default is None in which case it is
        2 * len(encodings).

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding

    """
    size = size if size is not None else 2 * len(encodings)
    fitness_arr = np.array(map(lambda encoding: encoding.fitness, encodings))
    for i in range(size // 2):
        yield tuple(np.random.choice(encodings, size=2, p=fitness_arr))
