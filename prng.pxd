cdef extern from "limits.h":
        int INT_MAX

# in order for this to be called from multithreaded nogil code, these functions need to be cdef
cdef class PRNG:
    cdef unsigned int[::1] thread_states
    cdef inline int randint(PRNG self) nogil
    cdef inline float randfloat(PRNG self) nogil
