cimport cython

from libc.math cimport log, sqrt, pow
from libc.stdlib cimport rand, RAND_MAX

from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from heap cimport Heap

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
UTYPE = np.uint64
ctypedef np.uint64_t UTYPE_t

ctypedef DTYPE_t (*CONFIDENCE_BOUND_FUN_t)(UTYPE_t, DTYPE_t, UTYPE_t, DTYPE_t)

cdef inline DTYPE_t alpha_alternative(UTYPE_t u, DTYPE_t delta, UTYPE_t n, DTYPE_t alpha_c):
    return sqrt(alpha_c * log(1 + (1 + log(u)) * n / delta) / u)

# alpha_c is unused in the theoretical bound
cdef inline DTYPE_t alpha_theoretical(UTYPE_t u, DTYPE_t delta, UTYPE_t n, DTYPE_t alpha_c):
    return sqrt(2 * beta(u, delta_prime(delta, n)) / u)

cdef inline DTYPE_t delta_prime(DTYPE_t delta, UTYPE_t n):
    return 1 - pow(1 - delta, 1.0 / n)

cdef inline DTYPE_t beta(DTYPE_t u, DTYPE_t delta):
    cdef DTYPE_t log_reciprocal_delta = -log(delta)
    return log_reciprocal_delta + 3 * log(log_reciprocal_delta) + 1.5 * log(1 + log(u))

cpdef inline UTYPE_t rand_upto(UTYPE_t n):
    return <UTYPE_t> (<DTYPE_t> rand() / (<DTYPE_t> RAND_MAX + 1.0) * n)

cdef inline bool lte(pair[DTYPE_t, UTYPE_t] a, pair[DTYPE_t, UTYPE_t] b):
    return a.first <= b.first

@cython.boundscheck(False)
@cython.wraparound(False)
cdef adaptive_knn(DTYPE_t[:] x, DTYPE_t[:, ::1] Y, UTYPE_t k, UTYPE_t h, DTYPE_t delta, str which_alpha, DTYPE_t alpha_c):

    cdef UTYPE_t n = Y.shape[0]
    cdef UTYPE_t m = Y.shape[1]

    cdef CONFIDENCE_BOUND_FUN_t alpha
    if which_alpha == 'theoretical':
        alpha = alpha_theoretical
    else:
        alpha = alpha_alternative


    if k + h >= n:
        raise ValueError('Requesting too many neighbors (k + h >= n).')

    # initialize the distance estimates

    cdef DTYPE_t[:] ds = np.empty(n, dtype=DTYPE)

    cdef UTYPE_t i, j

    for i in range(n):
        j = rand_upto(m)
        ds[i] = (x[j] - Y[i, j]) ** 2

    cdef UTYPE_t[:] ts = np.ones(n, dtype=UTYPE)

    # sort into heaps, as follows:
    # - three confidence heaps (top k max heap, next top h max heap, remaining min heap)
    # - four distance heaps (top k max heap, next top h min and max heaps, remaining min heap)

    cdef UTYPE_t[:] ds_sort_idx = np.argsort(ds).astype(UTYPE)

    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_top_v = vector[pair[DTYPE_t, UTYPE_t]](k)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_top_dist_v = vector[pair[DTYPE_t, UTYPE_t]](k)
    for i in range(k):
        j = ds_sort_idx[i]
        heap_top_v[i] = pair[DTYPE_t, UTYPE_t](ds[j] + alpha(ts[j], delta, n, alpha_c), j)
        heap_top_dist_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
    cdef Heap heap_top = Heap(heap_top_v, is_max_heap=True)
    cdef Heap heap_top_dist = Heap(heap_top_dist_v, is_max_heap=True)

    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_middle_v = vector[pair[DTYPE_t, UTYPE_t]](h)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_middle_dist_min_v = vector[pair[DTYPE_t, UTYPE_t]](h)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_middle_dist_max_v = vector[pair[DTYPE_t, UTYPE_t]](h)
    for i in range(h):
        j = ds_sort_idx[i + k]
        heap_middle_v[i] = pair[DTYPE_t, UTYPE_t](alpha(ts[j], delta, n, alpha_c), j)
        heap_middle_dist_min_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
        heap_middle_dist_max_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
    cdef Heap heap_middle = Heap(heap_middle_v, is_max_heap=True)
    cdef Heap heap_middle_dist_min = Heap(heap_middle_dist_min_v, is_max_heap=False)
    cdef Heap heap_middle_dist_max = Heap(heap_middle_dist_max_v, is_max_heap=True)

    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_bottom_v = vector[pair[DTYPE_t, UTYPE_t]](n - k - h)
    cdef vector[pair[DTYPE_t, UTYPE_t]] heap_bottom_dist_v = vector[pair[DTYPE_t, UTYPE_t]](n - k - h)
    for i in range(n - k - h):
        j = ds_sort_idx[i + k + h]
        heap_bottom_v[i] = pair[DTYPE_t, UTYPE_t](ds[j] - alpha(ts[j], delta, n, alpha_c), j)
        heap_bottom_dist_v[i] = pair[DTYPE_t, UTYPE_t](ds[j], j)
    cdef Heap heap_bottom = Heap(heap_bottom_v, is_max_heap=False)
    cdef Heap heap_bottom_dist = Heap(heap_bottom_dist_v, is_max_heap=False)

    # main loop

    cdef UTYPE_t t
    cdef pair[DTYPE_t, UTYPE_t] q1, q2, b1, b2, temp
    cdef DTYPE_t alpha_i, alpha_j
    cdef bool heaps_sorted

    for t in range(n * m):

        q1 = heap_top.head()
        b1 = heap_middle.head()
        q2 = heap_bottom.head()

        # terminating criterion

        if q1.first <= q2.first:
            break

        # update top k with least confidence

        i = q1.second
        alpha_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, alpha, alpha_c)
        heap_top.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i] + alpha_i, i))
        heap_top_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))

        ##########

        # update next_h

        if b1.first > alpha(ts[q2.second], delta, n, alpha_c):

            i = b1.second
            alpha_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, alpha, alpha_c)
            heap_middle.update_by_key(i, pair[DTYPE_t, UTYPE_t](alpha_i, i))
            heap_middle_dist_min.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))
            heap_middle_dist_max.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))

        # update remaining

        else:

            i = q2.second
            alpha_i = update_estimate_get_confidence(x, Y, ds, ts, i, delta, alpha, alpha_c)
            heap_bottom.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i] - alpha_i, i))
            heap_bottom_dist.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[i], i))

        # sort heaps

        heaps_sorted = False

        while not heaps_sorted:

            heaps_sorted = True

            q1 = heap_top_dist.head()
            b1 = heap_middle_dist_min.head()

            if q1.first > b1.first:

                heaps_sorted = False

                i = q1.second
                j = b1.second

                q2 = heap_top.get_by_key(i)
                b2 = heap_middle.get_by_key(j)

                alpha_i = q2.first - ds[i]
                alpha_j = b2.first

                heap_top_dist.update_by_key(i, b1)
                heap_top.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j] + alpha_j, j))

                heap_middle_dist_min.update_by_key(j, q1)
                heap_middle_dist_max.update_by_key(j, q1)
                heap_middle.update_by_key(j, pair[DTYPE_t, UTYPE_t](alpha_i, i))

            q1 = heap_bottom_dist.head()
            b1 = heap_middle_dist_max.head()

            if q1.first < b1.first:

                heaps_sorted = False

                i = q1.second
                j = b1.second

                q2 = heap_bottom.get_by_key(i)
                b2 = heap_middle.get_by_key(j)

                alpha_i = ds[i] - q2.first
                alpha_j = b2.first

                heap_bottom_dist.update_by_key(i, b1)
                heap_bottom.update_by_key(i, pair[DTYPE_t, UTYPE_t](ds[j] - alpha_j, j))

                heap_middle_dist_min.update_by_key(j, q1)
                heap_middle_dist_max.update_by_key(j, q1)
                heap_middle.update_by_key(j, pair[DTYPE_t, UTYPE_t](alpha_i, i))

    return heap_top_dist.v, heap_middle_dist_min.v, t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t update_estimate_get_confidence(DTYPE_t[:] x, DTYPE_t[:, ::1] Y, DTYPE_t[:] ds, UTYPE_t[:] ts, UTYPE_t i, DTYPE_t delta, CONFIDENCE_BOUND_FUN_t alpha, DTYPE_t alpha_c):

    cdef UTYPE_t n = Y.shape[0]
    cdef UTYPE_t m = Y.shape[1]

    if ts[i] < m:

        ts[i] += 1

        if ts[i] == m:

            ds[i] = distance2(x, Y, i) / m
            return 0

        else:

            j = rand_upto(m)
            ds[i] = (ds[i] * (ts[i] - 1)) / ts[i] + ((x[j] - Y[i, j]) ** 2) / ts[i]
            return alpha(ts[i], delta, n, alpha_c)

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t distance2(DTYPE_t[:] x, DTYPE_t[:, ::1] Y, UTYPE_t i):

    cdef DTYPE_t s = 0
    cdef UTYPE_t j
    for j in range(Y.shape[1]):
        s += (x[j] - Y[i, j]) ** 2

    return s


class AdaptiveKNN(object):

    def __init__(self, Y, k, h, delta=1e-6, which_alpha='theoretical', alpha_c=0.75):

        self.Y = Y
        self.k = k
        self.h = h
        self.delta = delta
        self.which_alpha = which_alpha
        self.alpha_c = alpha_c

    def query(self, x):

        top_k, next_h, t = adaptive_knn(x, self.Y, self.k, self.h, self.delta, self.which_alpha, self.alpha_c)
        return sorted(top_k), sorted(next_h), t
