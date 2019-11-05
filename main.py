import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2    # Убедиться, что x - двумерный тензор Numpy

    x = x.copy()    # Исключить затирание исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# Сложение двумерных тензоров с идентичными формами
def naive_add(x, y):
    assert len(x.shape) == 2    # Убедиться, что x и y — двумерные тензоры Numpy
    assert x.shape == y.shape

    x = x.copy()    # Исключить затирание исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


# Сложение дмумерного тензора с вектором
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2.   # Убедиться, что x — двумерный тензор Numpy.
    assert len(y.shape) == 1.   # Убедиться, что y — вектор Numpy.
    assert x.shape[1] == y.shape[0]

    x = x.copy()    # Исключить затирание исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


# Скалярное произведение двух векторов
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1    # Убедиться, что x и y — векторы Numpy
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# Скалаярное произведение матрицы на вектор
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2            # Убедиться, что x — матрица Numpy
    assert len(y.shape) == 1            # Убедиться, что y — вектор Numpy
    assert x.shape[1] == y.shape[0]     # Первое измерение x должно совпадать с нулевым измерением y!

    z = np.zeros(x.shape[0])            # Эта операция вернет вектор с нулевыми элементами, имеющий ту же форму, что и y
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


# Скалаярное произведение матрицы на вектор
def naive_matrix_vector_dot1(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


# Скалярное произведение двух матриц
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2            # Убедиться, что x и y — матрицы Numpy
    assert x.shape[1] == y.shape[0]     # Первое измерение x должно совпадать с нулевым измерением y!

    z = np.zeros((x.shape[0], y.shape[1]))  # Эта операция вернет матрицу заданной формы с нулевыми элементами
    for i in range(x.shape[0]):             # Обход строк в x...
        for j in range(y.shape[1]):         # ... и столбцов в y
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

