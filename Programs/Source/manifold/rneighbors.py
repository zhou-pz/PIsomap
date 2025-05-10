from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
import numpy as np

def array_sum(arr):
    size = len(arr)
    res = sfix(0)
    for i in range(size):
        res += arr[i]
    return res

def rneighbors(X, radius):
    n = X.shape[0]
    G = sfix.Matrix(n, n)   # nerghbor matrix
    inf = sfix(10000)

    for i in range(n - 1):
        for j in range(i + 1, n):
            temp = X[i] - X[j]
            d =  mpc_math.sqrt(array_sum(Array.create_from(temp * temp)))
            G[i][j] = (d > radius).if_else(inf, d)
    G = G + G.transpose()
    return G

def test_rneighbors():
    n = 50  # number of samples
    m = 8    # dimensions
    radius = 120
    np.random.seed(10)
    arr = np.random.randint(0, 5, size=(n, m))
    # print(arr)
    X = sfix.input_tensor_via(0, arr)   # input dataset
    G = rneighbors(X, radius)

