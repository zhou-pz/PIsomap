from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
from Programs.Source.manifold.topk import *

def array_sum(arr):
    size = len(arr)
    res = sfix(0)
    for i in range(size):
        res += arr[i]
    return res


def kneighbors(X, k):
    n = X.shape[0]
    G = sfix.Matrix(n, n)   # nerghbor matrix
    inf = sfix(10000)
    vinf = sfix.Array(n - k)
    vinf.assign_all(inf)

    for i in range(n - 1):
        for j in range(i + 1, n):
            temp = X[i] - X[j]
            G[i][j] = mpc_math.sqrt(array_sum(Array.create_from(temp * temp)))
    G = G + G.transpose()

    # #
    # print_ln("G:\n")
    # for i in range(n):
    #     G[i].print_reveal_nested(end='\n')

    @for_range_opt(n)
    def _(i):
    # for i in range(n):  # ************** 这个for循环展不开，轮数高
        pi = topk(G[i], k) # 附加logn比特没做
        G[i].assign(vinf, k)
        reveal_sort(pi, G[i])   # permute back
    # pi = topk.topk(G[0], k)
    
    # print_ln("G:\n")
    # for i in range(n):
    #     G[i].print_reveal_nested(end='\n')

    # print_ln("G:\n")
    # @for_range_opt(n)
    # def _(i):
    #     print_ln("%s",G[i].reveal())
    # Symmetrize the neighbor matrix.
    for i in range(n - 1):
        for j in range(i + 1, n):
            mini = G[i][j].min(G[j][i])
            G[i][j], G[j][i] = mini, mini

    return G


def kneighbors_baseline(X, k):
    n = X.shape[0]
    G = sfix.Matrix(n, n)   # nerghbor matrix
    inf = sfix(10000)
    vinf = sfix.Array(n - k)
    vinf.assign_all(inf)

    for i in range(n - 1):
        for j in range(i + 1, n):
            temp = X[i] - X[j]
            G[i][j] = mpc_math.sqrt(array_sum(Array.create_from(temp * temp)))
    G = G + G.transpose()
    # for i in range(n):
    #     for j in range(n):
    #         temp = X[i] - X[j]
    #         G[i][j] = mpc_math.sqrt(array_sum(Array.create_from(temp * temp)))

    @for_range_opt(n)
    def _(i):
        pi = topk(G[i], k) # 附加logn比特没做
        G[i].assign(vinf, k)
        reveal_sort(pi, G[i])   # permute back

    # Symmetrize the neighbor matrix.
    for i in range(n - 1):
        for j in range(i + 1, n):
            mini = G[i][j].min(G[j][i])
            G[i][j], G[j][i] = mini, mini
    return G