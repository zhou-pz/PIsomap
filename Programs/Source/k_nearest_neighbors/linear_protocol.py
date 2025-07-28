from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
from Programs.Source.manifold.topk import *


def array_sum(arr):
    size = len(arr)
    res = sfix(0)
    @for_range(size)
    def _(i):
        res.iadd(arr[i])
    return res


def knn(dataset, query, k):
    n = dataset.shape[0]
    dists = sfix.Array(n)
    temp = sfix.Array(128)

    for i in range(n):
        temp.assign((dataset[i] - query))

    # for i in range(n):
    #     temp.assign((dataset[i] - query).square())
    #     dists[i] = array_sum(temp)

    # pi = topk(dists, k) # 附加logn比特没做
    return dists
