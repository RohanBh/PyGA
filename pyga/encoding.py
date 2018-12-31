"""The :mod:`~pyga.encoding` provides an interface to create your own encoding or use any
of the existing ones."""
import abc
import functools

import numpy as np


@functools.total_ordering
class Encoding(abc.ABC):

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return len(self.value)

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
