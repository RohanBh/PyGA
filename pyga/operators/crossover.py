"""The :mod:`~pyga.operators.crossover` contains crossover operators"""
import operator

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
    index_list = list(map(operator.itemgetter(0), this.ivalue))
    that.ivalue = sorted(that.ivalue, key=lambda pair: index_list.index(pair[0]))
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


def two_point(parents, p):
    """Use :func:`point <pyga.operators.crossover.two_point>` to do a two point crossover between parents. The
        crossover point is selected randomly. Inversion crossover uses a master slave approach.

        Args:
            parents (tuple[pyga.encoding.BinaryEncoding]): The parents that produce offsprings.
            p (float): The probability with which crossover occurs.

        Returns:
            tuple(pyga.encoding.BinaryEncoding): Two offsprings.

        """
    if np.random.rand() <= p:
        p1, p2 = np.random.randint(1, len(parents), 2)
        p1, p2 = (p1, p2) if p1 < p2 else (p2, p1)
        value1 = parents[0].value[:p1] + parents[1].value[p1:p2] + parents[0].value[p2:]
        value2 = parents[1].value[:p1] + parents[0].value[p1:p2] + parents[1].value[p2:]
        return en.BinaryEncoding(value1), en.BinaryEncoding(value2)
    return parents


def multi_point(parents, p):
    """Use "func"`~pyga.operators.crossover.multi_point` to do a multi point crossover between parents.

    Args:
        parents (tuple[pyga.encoding.EvolvingHotspotEncoding]): The parents that produce offsprings.
        crossover_template (list[bool]): The boolean array which indicates where crossover must occur.
        p (float): The probability with which crossover occurs.

    Returns:
        tuple(pyga.encoding.EvolvingHotspotEncoding): Two offsprings.
    """
    if np.random.rand() <= p:
        value1 = []
        value2 = []
        template1 = []
        template2 = []
        crossover_state = False
        for t1, t2, v1, v2 in zip(parents[0].crossover_template, parents[1].crossover_template,
                                  parents[0].value, parents[1].value):
            is_crossover_point = t1
            crossover_state ^= is_crossover_point
            v1, v2, t1, t2 = (v1, v2, t1, t2) if not crossover_state else (v2, v1, t2, t1)
            value1.append(v1)
            value2.append(v2)
            template1.append(t1)
            template2.append(t2)
        return en.EvolvingHotspotEncoding(value1, template1), en.EvolvingHotspotEncoding(value1, template1)
    return parents


def uniform(parents, ph, p):
    """A uniform crossover between two parents is one in which an independent coin flip decides the fate of each allele.

    Args:
        parents (tuple[pyga.encoding.BinaryEncoding]): The parents that produce offsprings.
        ph (float): The parameter of the coin flip (h stands for heads). Typical values of ph are in range 0.5 - 0.8.
        p (float): The probability with which crossover occurs.

    Returns:
        tuple(pyga.encoding.BinaryEncoding): Two offsprings.
    """
    if np.random.rand() <= p:
        value1 = []
        value2 = []
        for v1, v2 in zip(parents[0].value, parents[1].value):
            v1, v2 = (v1, v2) if np.random.rand() <= ph else (v2, v1)
            value1.append(v1)
            value2.append(v2)
        return en.BinaryEncoding(value1), en.BinaryEncoding(value2)
    return parents
