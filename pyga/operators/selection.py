"""The :mod:`~pyga.operators.selection` contains different selection strategies."""
import operator

import numpy as np


def _default(encodings, size):
    size = size if size is not None else len(encodings)
    fitness_arr = np.array(map(operator.attrgetter('fitness'), encodings))
    return size, fitness_arr


def roulette_wheel(encodings, size=None):
    """A roulette wheel selection generator where each encoding is selected w.r.t its fitness.

    Args:
        encodings (list[pyga.encoding.Encoding]): Possible parents to select from.
        size (Optional[int]): The number of offsprings in next generation.
        Default is None in which case it is len(encodings).

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding

    """
    size, fitness_arr = _default(encodings, size)
    for i in range(size // 2):
        yield tuple(np.random.choice(encodings, size=2, p=fitness_arr))


def stochastic_universal_sampling(encodings, size=None):
    """Rather than spinning the wheel N times, SUS spins the wheel one time and uses N evenly spaced pointers to do the
    selection. This prevents premature convergence and allows for further exploration.

    Args:
        encodings (list[pyga.encoding.Encoding]): Parents to select from.
        size (int): Number of offsprings in the next generation.

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding
    """
    np.random.shuffle(encodings)
    size, fitness_arr = _default(encodings, size)
    # distance between pointers
    dist = fitness_arr.sum() / size
    ptr = np.random.rand() * dist
    cum_sum = 0
    prev = None
    for encoding, fitness in zip(encodings, fitness_arr):
        cum_sum += fitness
        if cum_sum > ptr:
            if prev is None:
                prev = encoding
            else:
                yield prev, encoding
                prev = None


def sigma_scaling(encodings, size=None):
    """This strategy keeps the selection pressure (that depends on fitness variance) relatively constant during the
    entire run of GA. It avoids premature convergence by relaxing the pressure when the variance is high in the
    beginning of the run. It also puts the pressure when the fitnesses of individuals are more or less the same
    allowing them to evolve more.

    For current implementation, probability of ith individual being selected in next generation is proportional to
    1 + (f(i) - f_mean(i))/(2*std_deviation).

    Args:
        encodings (list[pyga.encoding.Encoding]: Parents
        size (int): Number of offsprings

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding
    """
    size, fitness_arr = _default(encodings, size)
    mean, std_dev = fitness_arr.mean(), fitness_arr.std()
    weights = 1 + (fitness_arr - mean) / (2 * std_dev) if std_dev != 0 else 1
    weights[weights < 0] = 0.07
    for i in range(size // 2):
        yield tuple(np.random.choice(encodings, size=2, p=weights))


def elitism(encodings, size=1):
    """Elitism approach in genetic algorithm preserves some of the best individuals at each generation. It is a highly
    'exploitative' strategy. With this approach, we can guarantee that the GA doesn't waste its resources on
    rediscovering previously found good solutions.

    Args:
        encodings (list[pyga.encoding.Encoding]: The probable parents.
        size (int): Number of good solutions to preserve. Default is 1.

    Returns:
        The best "size" individuals.
    """
    _, fitness_arr = _default(encodings, size)
    indices = np.argpartition(fitness_arr, -size)[-size:]
    if type(encodings) == list: encodings = np.array(encodings)
    return encodings[indices]


def boltzmann_selection(encodings, mapping, size=None):
    """Boltzmann selection is similar to simulated annealing. The parameter temperature controls the selection pressure.
    Initially, the temperature is high, which is gradually lowered, thereby increasing the selection pressure and allowing
    GA to converge on best parts of the search space.

    For current implementation, probability of ith individual being selected in next generation is proportional to
    e ** (f(i) / mapping(i)).

    Args:
        encodings (list[pyga.encoding.Encoding]: The possible parents.
        mapping: The time (iteration) to temperature mapping.
        size (int): Number of offsprings needed.

    Notes:
        Too high value of fitnesses might cause the weights (probability of being in next generation) to go out of bound.

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding
    """
    size, fitness_arr = _default(encodings, size)
    for i in range(size // 2):
        temperature = mapping(T)
        weights = np.exp(fitness_arr / temperature)
        yield tuple(np.random.choice(encodings, size=2, p=weights))
