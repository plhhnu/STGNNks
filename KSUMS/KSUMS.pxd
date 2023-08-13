from libcpp.vector cimport vector

from Keep_order cimport Keep_order

cdef extern from "my.cpp":
    pass

cdef extern from "my.h":
    void argsort_f(int *v, int n, int *ind)
    void symmetry(vector[vector[int]] &NN, vector[vector[double]] &NND, double fill_ele);

cdef extern from "KSUMS.cpp":
    pass

cdef extern from "KSUMS.h":
    cdef cppclass KSUMS:
        int N
        int knn
        int c_true
        vector[vector[int]] NN
        vector[vector[double]] NND
        vector[int] y

        int *hi
        int *hi_TF
        int *hi_count

        int *knn_c
        int num_iter
        double max_d
        double _time

        Keep_order KO

        KSUMS() except +
        KSUMS(vector[vector[int]]& NN, vector[vector[double]]& NND, int c_true) except +
        void opt()
        void construct_hi(int sam_i)
