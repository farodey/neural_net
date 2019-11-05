import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2    # Убедиться, что x - двумерный тензор Numpy

    x = x.copy()    # Исключить затирание исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2.   # Убедиться, что x — двумерный тензор Numpy.
    assert len(y.shape) == 1.   # Убедиться, что y — вектор Numpy.
    assert x.shape[1] == y.shape[0]

    x = x.copy()    # Исключить затирание исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1    # Убедиться, что x и y — векторы Numpy
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


