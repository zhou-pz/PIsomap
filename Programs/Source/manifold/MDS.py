from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
from Programs.Source.manifold.jacobi import jacobi
from Programs.Source.manifold.topk import topk
import numpy as np
# program.use_trunc_pr = True # type: ignore

def array_sum(arr):
    size = len(arr)
    res = sfix(0)
    for i in range(size):
        res += arr[i]
    return res

def mds(D, t):
    n = D.shape[0]
    D_squa = D.same_shape()
    Y = sfix.Matrix(n, t)   # nerghbor matrix

    for i in range(n - 1):
        for j in range(i + 1, n):
            D_squa[i][j] = D[i][j].square()
    D_squa = D_squa + D_squa.transpose()
    
    # J = D.same_shape()
    # @for_range(n)
    # def _(i):
    #     J[i][i] = sfix(1)

    # cons1 = J.same_shape()
    # cons1.assign_all(1/n)
    # J = J - cons1

    # cons2 = J.same_shape()
    # cons2.assign_all(-0.5)

    # B = J.dot(D_squa).dot(J)
    # B.assign(cons2[:] * B[:])

    # print_ln("B:")
    # for i in range(n):
    #     B[i].print_reveal_nested(end='\n')

    B = D.same_shape()
    B.assign(-0.5 * D_squa[:])
    
    # Sum by row and /n
    dist_2_i = sfix.Array(n)
    for i in range(n):
        dist_2_i[i] = array_sum(B[i]) / n
    
    dist_2 = array_sum(dist_2_i) / n

    for i in range(n):
        for j in range(n):
            B[i][j] = B[i][j] - dist_2_i[i] - dist_2_i[j] + dist_2

    # print_ln("B:")
    # for i in range(n):
    #     B[i].print_reveal_nested(end='\n')

    eigenval, eigenvec = jacobi(B)

    # print_ln("eigenval:%s\n", eigenval.reveal())
    # print_ln("eigenvec:")
    # for i in range(eigenvec.shape[0]):
    #     eigenvec[i].print_reveal_nested(end='\n')


    pi = topk(eigenval, n - t) # permute smallest n - t values to left, then biggest t values are in the right  
    # print_ln("pi:%s\n", pi.reveal())
    reveal_sort(pi, eigenvec, reverse=True)
    eigenval_t = mpc_math.sqrt(eigenval[n - t : n])
    eigenvec_t = eigenvec.get_part(n - t, t).transpose()   

    # print_ln("tpok后：eigenvec_t:")
    # for i in range(eigenvec_t.shape[0]):
    #     eigenvec_t[i].print_reveal_nested(end='\n')

    eigenval_matr = sfix.Matrix(t, t)
    for i in range(t):
        eigenval_matr[i][i] = eigenval_t[i]
    Y.assign(eigenvec_t.dot(eigenval_matr))
    return Y

def mds_corr(D, t):
    n = D.shape[0]
    D_squa = D.same_shape()
    Y = sfix.Matrix(n, t)   # nerghbor matrix

    for i in range(n - 1):
        for j in range(i + 1, n):
            D_squa[i][j] = D[i][j].square()
    D_squa = D_squa + D_squa.transpose()
    B = D.same_shape()
    B.assign(-0.5 * D_squa[:])
    
    # Sum by row and /n
    dist_2_i = sfix.Array(n)
    for i in range(n):
        dist_2_i[i] = array_sum(B[i]) / n
    dist_2 = array_sum(dist_2_i) / n
    for i in range(n):
        for j in range(n):
            B[i][j] = B[i][j] - dist_2_i[i] - dist_2_i[j] + dist_2
            
    # print_ln("B:")
    # for i in range(n):
    #     B[i].print_reveal_nested(end='\n')

    eigenval, eigenvec = jacobi(B)

    pi = topk(eigenval, n - t) # permute smallest n - t values to left, then biggest t values are in the right  
    reveal_sort(pi, eigenvec, reverse=True)

    # print_ln("pi:")
    # pi.print_reveal_nested(end='\n')

    # print_ln("eigenval:")
    # eigenval.print_reveal_nested(end='\n')

    # print_ln("tpok后：eigenvec:")
    # for i in range(eigenvec.shape[0]):
    #     eigenvec[i].print_reveal_nested(end='\n')
        
    eigenval_t = mpc_math.sqrt(eigenval[n - t : n])
    eigenvec_t = eigenvec.get_part(n - t, t).transpose()   

    # print_ln("eigenvec_t:")
    # for i in range(eigenvec_t.shape[0]):
    #     eigenvec_t[i].print_reveal_nested(end='\n')

    eigenval_matr = sfix.Matrix(t, t)
    for i in range(t):
        eigenval_matr[i][i] = eigenval_t[i]
    Y.assign(eigenvec_t.dot(eigenval_matr))
    return Y, eigenval


def test_mds():
    n = 5  # number of samples
    t = 3   # reduced dimension
    np.random.seed(10)
    D = np.random.randint(0, 5, size=(n, n))
    # print(D)
    D = sfix.input_tensor_via(0, D)   # input dataset

    mds(D, t)

