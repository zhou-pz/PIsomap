from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
import numpy as np
import jax.numpy as jnp
# program.use_trunc_pr = True # type: ignore

MAX_ITER = 100

def array_sum(arr):
    size = len(arr)
    res = sfix(0)
    for i in range(size):
        res += arr[i]
    return res

def eye(n):
    I = sfix.Matrix(n, n)
    @for_range(n)
    def _(i):
        I[i][i] = sfix(1)
    return I

def elementSelection(n, m):
    # compute some cleartext indexes
    qq = []
    pp = []
    for k in range(1, 2*m):
        if k < m:
            q = np.arange(m - k + 1, n - k + 1)
            p = np.zeros_like(q)
            for i in range(len(q)):
                if q[i] <= 2*m - 2*k:
                    p[i] = (2*m - 2*k + 1) - q[i]
                elif 2*m - 2*k < q[i] <= 2*m - k - 1:
                    p[i] = (4*m - 2*k) - q[i]
                else:
                    p[i] = n
        else:
            q = np.arange(4*m - n - k, 3*m - k)
            p = np.zeros_like(q)
            for i in range(len(q)):
                if q[i] < 2*m - k + 1:
                    p[i] = n
                elif 2*m - k + 1 <= q[i] <= 4*m - 2*k - 1:
                    p[i] = (4*m - 2*k) - q[i]
                else:
                    p[i] = (6*m - 2*k - 1) - q[i]
        q -= 1
        p -= 1
        qq.append(q.tolist())
        pp.append(p.tolist())
    return pp, qq

def jacobi(A, tole = 1 * 1e-2):
    n = A.shape[0]
    m = (n + 1) // 2
    Q = sfix.Matrix(n, n)
    Q.assign(eye(n))
    p_, q_ = elementSelection(n, m) # compute some cleartext indexes
    count = regint(0)
    p = cint.Matrix(len(p_), len(p_[0]))
    q = cint.Matrix(len(q_), len(q_[0]))
    p.assign(p_)
    q.assign(q_)

    @for_range(MAX_ITER)
    def _(_):
        count.iadd(1)
        for k in range(2*m - 1):
            J = sfix.Matrix(n, n)
            J.assign(eye(n))
            n_elems = len(p[k])
            tar_elems1 = sfix.Array(n_elems)
            tar_elems2 = sfix.Array(n_elems)
            tar_elems3 = sfix.Array(n_elems)
            @for_range(n_elems)
            def _(i):
                tar_elems1[i] = A[p[k][i]][q[k][i]]
                tar_elems2[i] = A[p[k][i]][p[k][i]]
                tar_elems3[i] = A[q[k][i]][q[k][i]]
            
            tar_diff = tar_elems2 - tar_elems3            
            cos2θ = abs(tar_diff) / mpc_math.sqrt(4 * tar_elems1 * tar_elems1 + tar_diff * tar_diff)
            cos2 = 0.5 + 0.5 * cos2θ
            sin2 = 0.5 - 0.5 * cos2θ
            θc = mpc_math.sqrt(cos2)
            θs = mpc_math.sqrt(sin2)
            flag_zero = tar_elems1 == 0
            cos = θc * (1 - flag_zero) + flag_zero
            sin = θs * (((tar_elems1 * tar_diff) > 0) * 2 - 1)
            for i in range(n_elems):
                J[p[k][i]][p[k][i]], J[q[k][i]][q[k][i]] = cos[i], cos[i]
                J[p[k][i]][q[k][i]], J[q[k][i]][p[k][i]] = -sin[i], sin[i]
            A.assign(J.transpose().dot(A).dot(J))
            Q.assign(J.transpose().dot(Q))

        """ check for convergence"""
        sum_col = sfix.Array(n)
        @for_range(n)
        def _(i):
            sum_col[i] = array_sum(abs(A[i][:]))
        sum_matr = array_sum(sum_col)
        sum_diag = array_sum(abs(A.diag()))
        sum_offdiag = sum_matr - sum_diag
        print_ln("sum_offdiag: %s", sum_offdiag.reveal())
        @if_((sum_offdiag < (n * (n - 1) * tole)).reveal())
        def _():
            break_loop()

    print_ln("A:")
    @for_range_opt(A.shape[0])
    def _(i):
        print_ln("%s",A[i].reveal())

    print_ln("number of iterations: %s", count)
    eigenvals = sfix.Array(n)
    eigenvals.assign(A.diag())
    return eigenvals, Q

def test_jacobi():
    n = 250
    np.random.seed(10)
    A = np.random.uniform(0, 1.6, size=(n, n))
    A = (A + A.T) / 2
    np.set_printoptions(linewidth=150)

    sfix.set_precision(20, 42)
    sA = sfix.input_tensor_via(0, A)
    eigenvals, eigenvecs = jacobi(sA)

test_jacobi()