import cython
from libc.stdio cimport printf
#from libc.math cimport round as c_round
import numpy as np


cpdef ceiling(var_python_double):
    cdef double var_c_double = var_python_double
    printf("Running ceiling on %f.",var_c_double)
    temp = round(var_c_double)
    if temp<=var_c_double:
        return temp+1.0
    else:
        return temp
