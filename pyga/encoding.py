"""The :mod:`~pyga.encoding` provides an interface to create your own encoding or use any
of the existing ones."""
import abc
import collections
import functools

import numpy as np


@functools.total_ordering
class Encoding(abc.ABC):
    """Abstract class for the chromosome representation of the problem.

    Attributes:
        value (list): list/np.array that is the representation of the individual.
        ivalue (list): list of (index, value) pairs useful for inversion.
    """
    def __init__(self, value):
        self._value = value
        self.ivalue = np.array(enumerate(value))

    def __len__(self):
        return len(self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.ivalue = [(i, value[i]) for i, _ in enumerate(self.ivalue)]

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

    def __init__(self, value):
        """Create a new Binary encoding from the values (list like).

        Args:
            value: A list/tuple/ndarray that represents the encoding for the individual.
        """
        if type(value) != np.ndarray:
            value = np.array(value)
        super().__init__(value)
        self.value = value
