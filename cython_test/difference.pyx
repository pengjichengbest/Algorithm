

import numpy as np
cimport numpy as np
import cython
from libc.string cimport memset



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:] get_result(data, str y_name, str x_name):
    cdef:
        float[:] result = np.zeros(data.shape[0], dtype=np.float32)
        int rec1[10]
        int rec2[10]
        int[:] x = data[x_name].values
        int[:] y = data[y_name].values
    memset(rec1, 0, sizeof(rec1))
    memset(rec2, 0, sizeof(rec2))
    calculate(x, y, rec1, rec2, result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calculate(int[:] x, int[:] y, int[10] rec1, int[10] rec2, float[:] result):
    cdef int row
    cdef int nrow = x.shape[0]
    for row from 0 <= row < nrow by 1:
        rec1[x[row]] += 1
        rec2[x[row]] += y[row]
    for row from 0 <= row < nrow by 1:
        result[row] = (rec2[x[row]] - y[row]) / (rec1[x[row]] - 1)









