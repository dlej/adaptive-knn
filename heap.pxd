from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.uint64_t UTYPE_t
ctypedef bool (*COMP_t)(pair[DTYPE_t, UTYPE_t], pair[DTYPE_t, UTYPE_t])

cdef class Heap:

    cdef public vector[pair[DTYPE_t, UTYPE_t]] v
    cdef UTYPE_t n
    cdef dict map
    cdef COMP_t comp

    cpdef pair[DTYPE_t, UTYPE_t] head(self)
    cpdef void update(self, UTYPE_t i, pair[DTYPE_t, UTYPE_t] new_value)
    cpdef void update_by_key(self, UTYPE_t i, pair[DTYPE_t, UTYPE_t] new_value)
    cpdef pair[DTYPE_t, UTYPE_t] get(self, UTYPE_t i)
    cpdef pair[DTYPE_t, UTYPE_t] get_by_key(self, UTYPE_t i)
    cpdef UTYPE_t get_loc(self, UTYPE_t i)
    cdef void _bubble_up(self, UTYPE_t i)
    cdef void _bubble_down(self, UTYPE_t i)
    cdef void _heapify(self)
    cdef void _swap(self, UTYPE_t i, UTYPE_t j)
