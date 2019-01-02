"""The :mod:`~pyga.operators.crossover` contains crossover operators"""
import collections

import numpy as np
import pyga.encoding as en


def _reorder(this, that):
    """Reorders the ivalue list of `that` to match with `this`.

    Args:
        this (pyga.encoding.Encoding):
        that (pyga.encoding.Encoding):

    Returns:
        pyga.encoding.Encoding: that
    """
    d = collections.OrderedDict(this.ivalue)
    that.ivalue = sorted(that.ivalue, key=lambda pair: list(d.keys()).index(pair[0]))
    return that


def single_point(parents, p, inversion=False):
    """Use :func:`point <pyga.operators.crossover.single_point>` to do a single point crossover between parents. The
    crossover point is selected randomly. Inversion crossover uses a master slave approach.

    Args:
        parents (tuple[pyga.encoding.BinaryEncoding]): The parents that produce offsprings.
        p (float): The probability with which crossover occurs.
        inversion (bool): Whether to inversion is being used or not. Default is False.

    Returns:
        tuple(pyga.encoding.BinaryEncoding): Two offsprings.

    """
    if np.random.rand() <= p:
        # crossover occurs
        crossover_point = np.random.randint(1, len(parents))
        if inversion:
            _reorder(parents[0], parents[1])
            value1 = parents[0].ivalue[:crossover_point] + parents[1].ivalue[crossover_point:]
            value2 = parents[1].ivalue[:crossover_point] + parents[1].ivalue[crossover_point:]
        else:
            value1 = parents[0].value[:crossover_point] + parents[1].value[crossover_point:]
            value2 = parents[1].value[:crossover_point] + parents[0].value[crossover_point:]
        offspring1 = en.BinaryEncoding(value1, inversion)
        offspring2 = en.BinaryEncoding(value2, inversion)
        return offspring1, offspring2
    return parents
