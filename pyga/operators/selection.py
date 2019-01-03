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
        temperature = mapping(i)
        weights = np.exp(fitness_arr / temperature)
        yield tuple(np.random.choice(encodings, size=2, p=weights))


def linear_rank_selection(encodings, max_expected_offspring=1.1):
    """Rank selection is a strategy that ranks the individuals based on their fitness and then select them based solely
    on their rank. It is an explorative strategy because it disregards the difference in fitness values. In this scheme,
    the probability that an individual with rank r is selected is equal to:
        Min + (Max - Min) * (r - 1) / (N - 1)
    where Max is the max expected offsprings of the individual with rank 1 and similarly Min is the expected offsprings
    of individual with rank N.
    By normalization (sum of expectations = N) and the constraint (Max > Min > 0), we have:
        1 <= Max <= 2 and Min = 2 - Max
    Rank selection leads to slower convergence (because of lower selection pressure), but usually this leads to
    more successful searches.
    SUS is used to sample parents after the ranks have been assigned.
    Args:
        encodings (list[pyga.encoding.Encoding]: Possible parents.
        max_expected_offspring (float): The expected number of offsprings of the individual with rank 1. Default is 1.1

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding
    """
    if not isinstance(encodings, np.ndarray): encodings = np.array(encodings)
    encodings = np.sort(encodings, order='fitness')[::-1]
    size = len(encodings)
    _min, _max = 2 - max_expected_offspring, max_expected_offspring
    weights = _min + (_max - _min) * (np.arange(1, size + 1) - 1) / (size - 1)
    for encoding, weight in zip(encodings, weights):
        encoding.fitness = weight
    return stochastic_universal_sampling(encodings)


def tournament_selection(encodings, selection_bias=0.75, size=None):
    """Similar to rank selection in terms of selection pressure, but computationally more efficient. It selects two
    individuals at random and has a match/battle between them (figuratively). Then, the fitter of the two is selected with
    probability "selection_bias". This selection is done with replacement.

    Args:
        encodings (list[pyga.encoding.Encoding]: Possible parents.
        selection_bias (float): The probability with which the fitter individual is selected. Default is 0.75
        size (int): Number of offsprings needed.

    Yields:
        tuple(pyga.encoding.Encoding): a pair of parents selected from encoding
    """
    size, _ = _default(encodings, size)
    prev = None
    for i in range(size // 2):
        en1, en2 = np.random.choice(encodings, size=2)
        en1, en2 = (en1, en2) if en1.fitness > en2.fitness else (en2, en1)
        selected = en1 if np.random.rand() < selection_bias else en2
        if prev is None:
            prev = selected
        else:
            yield prev, selected
            prev = None
