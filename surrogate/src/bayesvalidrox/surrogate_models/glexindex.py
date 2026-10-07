#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi indices for monomial exponents.
Credit: Jonathan Feinberg
https://github.com/jonathf/numpoly/blob/master/numpoly/utils/glexindex.py
"""

import numpy
import numpy.typing


def glexindex(
    start, stop=None, dimensions=1, cross_truncation=1.0, graded=False, reverse=False
):
    """
    Generate graded lexicographical multi-indices for the monomial exponents.
    Args:
        start (Union[int, numpy.ndarray]):
            The lower order of the indices. If array of int, counts as lower
            bound for each axis.
        stop (Union[int, numpy.ndarray, None]):
            The maximum shape included. If omitted: stop <- start; start <- 0
            If int is provided, set as largest total order. If array of int,
            set as upper bound for each axis.
        dimensions (int):
            The number of dimensions in the expansion.
        cross_truncation (float, Tuple[float, float]):
            Use hyperbolic cross truncation scheme to reduce the number of
            terms in expansion. If two values are provided, first is low bound
            truncation, while the latter upper bound. If only one value, upper
            bound is assumed.
        graded (bool):
            Graded sorting, meaning the indices are always sorted by the index
            sum. E.g. ``(2, 2, 2)`` has a sum of 6, and will therefore be
            consider larger than both ``(3, 1, 1)`` and ``(1, 1, 3)``.
        reverse (bool):
            Reversed lexicographical sorting meaning that ``(1, 3)`` is
            considered smaller than ``(3, 1)``, instead of the opposite.
    Returns:
        list:
            Order list of indices.
    Examples:
        >>> numpoly.glexindex(4).tolist()
        [[0], [1], [2], [3]]
        >>> numpoly.glexindex(2, dimensions=2).tolist()
        [[0, 0], [1, 0], [0, 1]]
        >>> numpoly.glexindex(start=2, stop=3, dimensions=2).tolist()
        [[2, 0], [1, 1], [0, 2]]
        >>> numpoly.glexindex([1, 2, 3]).tolist()
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]]
        >>> numpoly.glexindex([1, 2, 3], cross_truncation=numpy.inf).tolist()
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [0, 0, 2], [0, 1, 2]]
    """
    if stop is None:
        start, stop = 0, start
    start = numpy.array(start, dtype=int).flatten()
    stop = numpy.array(stop, dtype=int).flatten()
    start, stop, _ = numpy.broadcast_arrays(start, stop, numpy.empty(dimensions))

    cross_truncation = cross_truncation * numpy.ones(2)

    # Moved here from _glexindex
    bound = stop.max()
    dimensions = len(start)
    start = numpy.clip(start, a_min=0, a_max=None)
    dtype = numpy.uint8 if bound < 256 else numpy.uint16
    range_ = numpy.arange(bound, dtype=dtype)
    indices = range_[:, numpy.newaxis]

    for idx in range(dimensions - 1):

        # Truncate at each step to keep memory usage low
        if idx:
            indices = indices[cross_truncate(indices, bound - 1, cross_truncation[1])]

        # Repeats the current set of indices.
        # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
        indices = numpy.tile(indices, (bound, 1))

        # Stretches ranges over the new dimension.
        # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
        front = range_.repeat(len(indices) // bound)[:, numpy.newaxis]

        # Puts them two together.
        indices = numpy.column_stack((front, indices))

    # Complete the truncation scheme
    if dimensions == 1:
        indices = indices[(indices >= start) & (indices < bound)]
    else:
        lower = cross_truncate(indices, start - 1, cross_truncation[0])
        upper = cross_truncate(indices, stop - 1, cross_truncation[1])
        indices = indices[lower ^ upper]

    indices = numpy.array(indices, dtype=int).reshape(-1, dimensions)
    if indices.size:
        # moved here from glexsort
        keys = indices.T
        keys_ = numpy.atleast_2d(keys)
        if reverse:
            keys_ = keys_[::-1]

        indices_sort = numpy.array(numpy.lexsort(keys_))
        if graded:
            indices_sort = indices_sort[
                numpy.argsort(numpy.sum(keys_[:, indices_sort], axis=0))
            ].T

        indices = indices[indices_sort]
    return indices


def cross_truncate(indices, bound, norm):
    r"""
    Truncate of indices using L_p norm.
    .. math:
        L_p(x) = \sum_i |x_i/b_i|^p ^{1/p} \leq 1
    where :math:`b_i` are bounds that each :math:`x_i` should follow.
    Args:
        indices (Sequence[int]):
            Indices to be truncated.
        bound (int, Sequence[int]):
            The bound function for witch the indices can not be larger than.
        norm (float, Sequence[float]):
            The `p` in the `L_p`-norm. Support includes both `L_0` and `L_inf`.
    Returns:
        Boolean indices to ``indices`` with True for each index where the
        truncation criteria holds.
    Examples:
        >>> indices = numpy.array(numpy.mgrid[:10, :10]).reshape(2, -1).T
        >>> indices[cross_truncate(indices, 2, norm=0)].T
        array([[0, 0, 0, 1, 2],
               [0, 1, 2, 0, 0]])
        >>> indices[cross_truncate(indices, 2, norm=1)].T
        array([[0, 0, 0, 1, 1, 2],
               [0, 1, 2, 0, 1, 0]])
        >>> indices[cross_truncate(indices, [0, 1], norm=1)].T
        array([[0, 0],
               [0, 1]])
    """
    assert norm >= 0, "negative L_p norm not allowed"
    bound = numpy.asarray(bound, dtype=float).flatten() * numpy.ones(indices.shape[1])

    if numpy.any(bound < 0):
        return numpy.zeros((len(indices),), dtype=bool)

    if numpy.any(bound == 0):
        out = numpy.all(indices[:, bound == 0] == 0, axis=-1)
        if numpy.any(bound):
            out &= cross_truncate(indices[:, bound != 0], bound[bound != 0], norm=norm)
        return out

    if norm == 0:
        out = numpy.sum(indices > 0, axis=-1) <= 1
        out[numpy.any(indices > bound, axis=-1)] = False
    elif norm == numpy.inf:
        out = numpy.max(indices / bound, axis=-1) <= 1
    else:
        out = numpy.sum((indices / bound) ** norm, axis=-1) ** (1.0 / norm) <= 1

    assert numpy.all(out[numpy.all(indices == 0, axis=-1)])

    return out
