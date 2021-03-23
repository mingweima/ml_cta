import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
# from libc.stdio cimport printf
from cython.parallel import prange, parallel

ctypedef np.uint8_t uint8  # we use this for boolean type

__all__ = ['_fractal_filter', '_convert_label']


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _find_abs(const double* low, const double* high, uint8* abs,
                    int n) nogil:
    cdef size_t i;
    abs[0] = 0
    abs[n - 1] = 0
    for i in prange(1, n - 1, nogil=True, schedule='static'):
        if low[i] > low[i - 1] and high[i] < high[i - 1]:
            abs[i] = 1
        elif low[i] > low[i + 1] and high[i] < high[i + 1]:
            abs[i] = 1
        else:
            abs[i] = 0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _find_fractal(const double* low, const double* high, const int n,
                        const double th_a, const double th_b, const int n_suc,
                        uint8* t_j, uint8* b_j, uint8* t_l, uint8* t_r,
                        uint8* b_l, uint8* b_r) nogil:
    '''n >= 2'''
    cdef size_t i, j, k;

    # # 判断顶和底
    # for i in prange(1, n - 1, nogil=True, schedule='static'):
    #     if (low[i] - low[i - 1] > th_b and high[i] - high[i - 1] > th_a and
    #         low[i] - low[i + 1] > th_b and high[i] - high[i + 1] > th_a):
    #         t_j[i] = 1
    #     if (low[i - 1] - low[i] > th_a and high[i - 1] - high[i] > th_b and
    #         low[i + 1] - low[i] > th_a and high[i + 1] - high[i] > th_b):
    #         b_j[i] = 1

    # 判断顶和底
    for i in prange(1, n - 1, nogil=True, schedule='static'):
        if (low[i] - low[i - 1] > th_b and high[i] - high[i - 1] > th_a and
            low[i] - low[i + 1] >= th_b and high[i] - high[i + 1] > th_a):
            t_j[i] = 1
        if (low[i - 1] - low[i] > th_a and high[i - 1] - high[i] > th_b and
            low[i + 1] - low[i] > th_a and high[i + 1] - high[i] >= th_b):
            b_j[i] = 1

    # 紧连着的分形，最多只能有n_suc个
    for i in range(1, n - 1):
        if t_j[i] or b_j[i]:
            j = i
            for j in range(i + 1, n - 1):
                if (t_j[j] == 0 and b_j[j] == 0):
                    break
            if j - i > n_suc or j - i ==2:
                for k in range(i, j):
                    t_j[k] = 0
                    b_j[k] = 0

    # 标记顶和底的左右
    for i in range(1, n - 1):
        if t_j[i]:
            t_l[i - 1] = 1
            t_r[i + 1] = 1
        elif b_j[i]:
            b_l[i - 1] = 1
            b_r[i + 1] = 1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef int _get_unabs(double* low, double* high, double* low_2, double* high_2,
                    uint8* t_j, uint8* b_j, uint8* t_l, uint8* t_r, uint8* b_l,
                    uint8* b_r, uint8* abs, uint8* t_j_2, uint8* b_j_2,
                    uint8* t_l_2, uint8* t_r_2, uint8* b_l_2, uint8* b_r_2,
                    const int n) nogil:
    cdef size_t i, j = 0
    for i in prange(n, nogil=True, schedule='static'):
        if (abs[i] and not t_j[i] and not b_j[i] and not t_l[i] and not b_l[i]
            and not t_r[i] and not b_r[i]):
            abs[i] = 1
        else:
            abs[i] = 0
    for i in range(n):
        if not abs[i]:
            low_2[j] = low[i]
            high_2[j] = high[i]
            t_j_2[j] = t_j[i]
            t_l_2[j] = t_l[i]
            t_r_2[j] = t_r[i]
            b_j_2[j] = b_j[i]
            b_l_2[j] = b_l[i]
            b_r_2[j] = b_r[i]
            j += 1
    return j


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _merge_unabs(double* low, double* high, double* low_2, double* high_2,
                       uint8* t_j, uint8* b_j, uint8* t_l, uint8* t_r,
                       uint8* b_l, uint8* b_r, uint8* abs, uint8* t_j_2,
                       uint8* b_j_2, uint8* t_l_2, uint8* t_r_2, uint8* b_l_2,
                       uint8* b_r_2, const int n) nogil:
    cdef size_t i, j = 0
    for i in range(n):
        if not abs[i]:
            if t_j_2[j]:
                t_j[i] = 1
            if b_j_2[j]:
                b_j[i] = 1
            if t_l_2[j]:
                t_l[i] = 1
            if t_r_2[j]:
                t_r[i] = 1
            if b_l_2[j]:
                b_l[i] = 1
            if b_r_2[j]:
                b_r[i] = 1
            j += 1


# 0: 顶
# 1: 底
# 2: 顶左
# 3: 顶右
# 4: 底左
# 5: 底右
# 6: abs

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _fractal_filter(const double[::1] low, const double[::1] high,
                    uint8[:, ::1] label, const int n, const double th_a,
                    const double th_b, const int n_suc):
    cdef int m
    cdef double *low_2 = <double *> malloc(n * sizeof(double))
    cdef double *high_2 = <double *> malloc(n * sizeof(double))
    cdef uint8 *label_2 = <uint8 *> malloc(n * 6 * sizeof(uint8))
    if not low_2 or not high_2 or not label_2:
        raise MemoryError('cannot malloc required array in _fractal_filter.')
    try:
        # printf('0\n')
        _find_fractal(&low[0], &high[0], n, th_a, th_b, n_suc, &label[0, 0],
                      &label[1, 0], &label[2, 0], &label[3, 0], &label[4, 0],
                      &label[5, 0])
        # printf('1\n')
        _find_abs(&low[0], &high[0], &label[6, 0], n)
        # printf('2\n')
        m = _get_unabs(&low[0], &high[0], &low_2[0], &high_2[0], &label[0, 0],
                       &label[1, 0], &label[2, 0], &label[3, 0], &label[4, 0],
                       &label[5, 0], &label[6, 0], &label_2[0], &label_2[n],
                       &label_2[2 * n], &label_2[3 * n], &label_2[4 * n],
                       &label_2[5 * n], n)
        # printf('3\n')
        _find_fractal(&low_2[0], &high_2[0], m, th_a, th_b, n_suc, &label_2[0],
                      &label_2[n], &label_2[2 * n], &label_2[3 * n],
                      &label_2[4 * n], &label_2[5 * n])
        # printf('4\n')
        _merge_unabs(&low[0], &high[0], &low_2[0], &high_2[0], &label[0, 0],
                     &label[1, 0], &label[2, 0], &label[3, 0], &label[4, 0],
                     &label[5, 0], &label[6, 0], &label_2[0], &label_2[n],
                     &label_2[2 * n], &label_2[3 * n], &label_2[4 * n],
                     &label_2[5 * n], n)
        # printf('5\n')
    finally:
        free(low_2)
        free(high_2)
        free(label_2)


# -1: 无
# 0: 顶
# 1: 底
# 2: 顶左
# 3: 顶右
# 4: 底左
# 5: 底右
# 6: 顶abs
# 7: 底abs

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def _convert_label(const uint8[:, ::1] label, long[::1] label_2, const int n):
    cdef size_t i, j
    for i in range(n):
        if label[0, i]:
            label_2[i] = 0
            for j in range(i - 1, -1, -1):
                if label[2, j]:
                    if label_2[j] == -1:
                        label_2[j] = 2
                    break
                elif label[6, j]:
                    if label_2[j] == -1:
                        label_2[j] = 6
                elif label[1, j]:
                    break
                else:
                    raise RuntimeError('unexpected behavior 0, i={}, '
                                       'j={}.'.format(i, j))
                if j == 0:
                    raise RuntimeError('unexpected behavior 1, i={}, '
                                       'j={}.'.format(i, j))
            for j in range(i + 1, n):
                if label[3, j]:
                    if label_2[j] == -1:
                        label_2[j] = 3
                    break
                elif label[6, j]:
                    if label_2[j] == -1:
                        label_2[j] = 6
                elif label[1, j]:
                    break
                else:
                    raise RuntimeError('unexpected behavior 2, i={}, '
                                       'j={}.'.format(i, j))
                if j == n - 1:
                    raise RuntimeError('unexpected behavior 3, i={}, '
                                       'j={}.'.format(i, j))
    for i in range(n):
        if label[1, i]:
            label_2[i] = 1
            for j in range(i - 1, -1, -1):
                if label[4, j]:
                    if label_2[j] == -1:
                        label_2[j] = 4
                    break
                elif label[6, j]:
                    if label_2[j] == -1:
                        label_2[j] = 7
                elif label[0, j]:
                    break
                else:
                    raise RuntimeError('unexpected behavior 4, i={}, '
                                       'j={}.'.format(i, j))
                if j == 0:
                    raise RuntimeError('unexpected behavior 5, i={}, '
                                       'j={}.'.format(i, j))
            for j in range(i + 1, n):
                if label[5, j]:
                    if label_2[j] == -1:
                        label_2[j] = 5
                    break
                elif label[6, j]:
                    if label_2[j] == -1:
                        label_2[j] = 7
                elif label[0, j]:
                    break
                else:
                    raise RuntimeError('unexpected behavior 6, i={}, '
                                       'j={}.'.format(i, j))
                if j == n - 1:
                    raise RuntimeError('unexpected behavior 7, i={}, '
                                       'j={}.'.format(i, j))
