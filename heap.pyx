# cython: profile=True

from libcpp cimport bool
from libcpp.algorithm cimport make_heap, push_heap, pop_heap
from libcpp.vector cimport vector
from libcpp.utility cimport pair

from cython.view cimport array

cdef inline bool gte(pair[DTYPE_t, UTYPE_t] a, pair[DTYPE_t, UTYPE_t] b):
    return a.first >= b.first

cdef inline bool lte(pair[DTYPE_t, UTYPE_t] a, pair[DTYPE_t, UTYPE_t] b):
    return a.first <= b.first

cdef class Heap:

    def __init__(self, vector[pair[DTYPE_t, UTYPE_t]] v, bool is_max_heap=True):

        self.v = v
        self.n = v.size()

        self.map = {}

        cdef UTYPE_t i
        for i in range(self.n):

            self.map[self.v[i].second] = i

        if is_max_heap:
            self.comp = lte
        else:
            self.comp = gte

        self._heapify()

    cpdef pair[DTYPE_t, UTYPE_t] head(self):

        return self.get(0)

    cpdef void update(self, UTYPE_t i, pair[DTYPE_t, UTYPE_t] new_value):

        cdef pair[DTYPE_t, UTYPE_t] old_value = self.v[i]
        self.v[i] = new_value

        del self.map[old_value.second]
        self.map[new_value.second] = i

        if self.comp(new_value, old_value):
            self._bubble_down(i)
        else:
            self._bubble_up(i)

    cpdef void update_by_key(self, UTYPE_t i, pair[DTYPE_t, UTYPE_t] new_value):

        self.update(self.get_loc(i), new_value)

    cpdef pair[DTYPE_t, UTYPE_t] get(self, UTYPE_t i):

        return self.v[i]

    cpdef pair[DTYPE_t, UTYPE_t] get_by_key(self, UTYPE_t i):

        return self.get(self.get_loc(i))

    cpdef UTYPE_t get_loc(self, UTYPE_t i):

        return self.map[i]

    cdef void _bubble_up(self, UTYPE_t i):

        if i == 0:
            return

        cdef UTYPE_t parent = (i - 1) // 2

        if self.comp(self.v[parent], self.v[i]):
            self._swap(parent, i)
            self._bubble_up(parent)

    cdef void _bubble_down(self, UTYPE_t i):

        cdef UTYPE_t left = 2*i + 1
        cdef UTYPE_t right = 2*i + 2
        cdef UTYPE_t largest = i

        if left < self.n and self.comp(self.v[largest], self.v[left]):
            largest = left
        if right < self.n and self.comp(self.v[largest], self.v[right]):
            largest = right

        if largest != i:
            self._swap(i, largest)
            self._bubble_down(largest)

    cdef void _heapify(self):

        cdef UTYPE_t i
        for i in range((self.n // 2) - 1, -1, -1):
            self._bubble_down(i)

    cdef void _swap(self, UTYPE_t i, UTYPE_t j):

        cdef pair[DTYPE_t, UTYPE_t] temp = self.v[i]
        self.v[i] = self.v[j]
        self.v[j] = temp

        self.map[self.v[i].second] = i
        self.map[self.v[j].second] = j
