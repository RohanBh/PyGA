"""The :mod:`~pyga.encoding` provides an interface to create your own encoding or use any
of the existing ones."""
import abc
import functools
import operator

import numpy as np


@functools.total_ordering
class Encoding(abc.ABC):
    """Abstract class for the chromosome representation of the problem.

    Attributes:
        value (list): list/np.array that is the representation of the individual.
        ivalue (list): list of (index, value) pairs useful for inversion.
    """

    def __init__(self, value, is_ivalue=False):
        """Creates a new encoding.

        Args:
            value: list that represents the encoding.
            is_ivalue: If True, treat the list as :attr:`ivalue`. Default is False.
        """
        if is_ivalue:
            self.ivalue = value
            self.value = list(map(operator.itemgetter(1), sorted(value, key=operator.itemgetter(0))))
        else:
            self._value = value
            self.ivalue = list(enumerate(value))

    def __len__(self):
        return len(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.ivalue = [(i, value[i]) for i, _ in self.ivalue]

    @property
    @abc.abstractmethod
    def fitness(self):
        return NotImplemented

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness


class BinaryEncoding(Encoding, abc.ABC):
    """An :class:`~pyga.encoding.Encoding` that is represented by a bitstring. The bitstring is a numpy boolean
    array."""

    def __init__(self, value, is_ivalue=False):
        """Create a new Binary encoding from the values (list like).

        Args:
            value: A list/tuple/ndarray that represents the encoding for the individual.
            is_ivalue: If True, treat the list as :attr:`ivalue`. Default is False.
        """
        if type(value) != np.ndarray:
            value = np.array(value)
        super().__init__(value, is_ivalue)


class EvolvingHotspotEncoding(Encoding, abc.ABC):
    """An abstract class that uses "evolving crossover hotspots", i.e. it evolves the points where crossover occurs
    along with the solutions. The crossover hotspots are denoted by a separate boolean array. Note that if the encoding
    length is N then the length of crossover hotspot array will be N + 1.

    Examples:
        If :attr:`value` = [0,1,1,0] and crossover_template = [1,0,1,1,0], then crossover occurs at positions 0,
    """

    def __init__(self, value, crossover_template=None):
        super().__init__(value)
        if crossover_template is None:
            self.crossover_template = [False] * len(value)
        else:
            self.crossover_template = crossover_template
