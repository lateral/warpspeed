#cython: boundscheck=False, wraparound=False
from cython.parallel cimport prange
cimport openmp
import numpy as np

# Python visible constant
int_max = INT_MAX


cdef class PRNG:
    """
    Instances of this class provide pseudo-random number generation that is
    thread safe: the random results obtained by any thread depends only upon
    the initial seed for that thread, and not on the order in which the
    different threads draw.
    """

    def __init__(self, thread_seeds=None):
        if thread_seeds is None:
            # no seeds specified, so choose something random
            max_threads = openmp.omp_get_max_threads()
            self.thread_states = np.random.randint(0, INT_MAX, size=max_threads).astype(np.uint32)
        else:
            self.thread_states = thread_seeds

    cdef inline int randint(PRNG self) nogil:
        cdef:
            unsigned int x
            unsigned int thread_number = openmp.omp_get_thread_num()
            unsigned int and_1 = 0x9D2C5680
            unsigned int and_2 = 0xEFC60000
        self.thread_states[thread_number] = self.thread_states[thread_number] * 1103515245 + 12345
        x = self.thread_states[thread_number]
        x = x ^ (x >> 11)
        x = x ^ (x << 7 & and_1)
        x = x ^ (x << 15 & and_2)
        x = x ^ (x >> 18)
        return x / 2

    cdef inline float randfloat(PRNG self) nogil:
        """
        Returns a random float in the range 0 .. 1, updating the seed for the calling thread.
        """
        return self.randint() / float(INT_MAX)


### TESTING FUNCTIONS
# The following functions are intended for testing the thread-safety of the PRNG (from Python code)


cpdef randint_roundrobin(unsigned int[::1] seeds, int[::1] samples):
    """
    Populate the array `samples` with the result of multithreaded calls to
    randint(), where the drawings are done in a round robin fashion.  The
    number of threads used is determined by the length of the seeds array.
    """
    cdef:
        unsigned int num_threads = seeds.shape[0]
        PRNG prng = PRNG(seeds)
        int num_rows = samples.shape[0]
        int row

    for row in prange(num_rows, schedule='static', chunksize=1, nogil=True, num_threads=num_threads):
        samples[row] = prng.randint()


cpdef randfloat_roundrobin(unsigned int[::1] seeds, float[::1] samples):
    """
    As per randint_roundrobin, but for the randfloat function.
    """
    cdef:
        unsigned int num_threads = seeds.shape[0]
        PRNG prng = PRNG(seeds)
        int num_rows = samples.shape[0]
        int row

    for row in prange(num_rows, schedule='static', chunksize=1, nogil=True, num_threads=num_threads):
        samples[row] = prng.randfloat()
