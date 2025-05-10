from Compiler.library import *
from Compiler.sorting import *
from Compiler import mpc_math
import numpy as np
import jax.numpy as jnp

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
    '''
    More efficient element selection method (also the standard one).
    '''
    qq = []  # 存储每次迭代的q
    pp = []  # 存储每次迭代的p
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
        # for i in range(len(q)):
        #     if q[i] > p[i]:
        #         q[i], p[i] = p[i], q[i]
        q -= 1
        p -= 1
        qq.append(q.tolist())
        pp.append(p.tolist())
    return pp, qq

def elementSelection2(num_samples, m):
    qq = []  # 存储每次迭代的q
    pp = []  # 存储每次迭代的p
    for i in range(1, num_samples + (num_samples - 1) // 2):
        if i < num_samples:
            l_0 = i
            r_0 = 0
        else:
            l_0 = num_samples - 1
            r_0 = i - l_0
        
        n = (l_0 - r_0 - 1) // 2 + 1

        j_indices = jnp.arange(n)
        l = l_0 - j_indices
        r = r_0 + j_indices

        if i < num_samples // 2:
            l_1 = num_samples - 1 - r_0
            r_1 = num_samples - 1 - l_0
            n = (l_1 - r_1 - 1) // 2 + 1
            j_indices = jnp.arange(n)
            l_1 = l_1 - j_indices
            r_1 = r_1 + j_indices
            l = jnp.concatenate([l, l_1])
            r = jnp.concatenate([r, r_1])
        qq.append(r.tolist())
        pp.append(l.tolist())
    return pp, qq

def jacobi(A, tole = 1 * 1e-2):
    n = A.shape[0]
    m = (n + 1) // 2
    Q = sfix.Matrix(n, n)
    Q.assign(eye(n))
    p, q = elementSelection(n, m)
    count = regint(0)
    # p_, q_ = elementSelection2(n, m)
    # B = np.zeros((n, n))
    # for i in range(len(p)):
    #     for j in range(len(p[0])):
    #         B[p[i][j]][q[i][j]] += 1
    # print("sweeped place:\n", B)
    times = regint(3)
    # p = cint.Matrix(len(p_), len(p_[0]))
    # q = cint.Matrix(len(q_), len(q_[0]))
    # p.assign(p_)
    # q.assign(q_)

    @for_range(MAX_ITER)
    def _(_):
    # for _ in range(MAX_ITER): # elementSelection2
        count.iadd(1)
        for k in range(2*m - 1):
        # for k in range(n + (n - 1) // 2 - 1): # elementSelection2
            J = sfix.Matrix(n, n)
            J.assign(eye(n))
            n_elems = len(p[k]) # ********************
            tar_elems1 = sfix.Array(n_elems)
            tar_elems2 = sfix.Array(n_elems)
            tar_elems3 = sfix.Array(n_elems)
            # @for_range(n_elems)
            # def _(i): #******************
            for i in range(n_elems):
                tar_elems1[i] = A[p[k][i]][q[k][i]]
                tar_elems2[i] = A[p[k][i]][p[k][i]]
                tar_elems3[i] = A[q[k][i]][q[k][i]]
            tar_diff = tar_elems2 - tar_elems3            
            cos2θ = abs(tar_diff) / mpc_math.sqrt(4 * tar_elems1 * tar_elems1 + tar_diff * tar_diff)
            # cos2θ = abs(tar_diff) * mpc_math.InvertSqrt(4 * tar_elems1 * tar_elems1 + tar_diff * tar_diff)
            cos2 = 0.5 + 0.5 * cos2θ
            sin2 = 0.5 - 0.5 * cos2θ
            θc = mpc_math.sqrt(cos2)
            θs = mpc_math.sqrt(sin2)
            flag_zero = tar_elems1 == 0
            cos = θc * (1 - flag_zero) + flag_zero
            # sin = θs * (((tar_elems1 * tar_diff) > 0) * 2 - 1)   * (1 - (tar_elems1 == 0) * (tar_diff == 0))
            sin = θs * (((tar_elems1 * tar_diff) > 0) * 2 - 1)
            # @for_range(n_elems)
            # def _(i):
            for i in range(n_elems): #******************
                J[p[k][i]][p[k][i]], J[q[k][i]][q[k][i]] = cos[i], cos[i]
                J[p[k][i]][q[k][i]], J[q[k][i]][p[k][i]] = -sin[i], sin[i]

            # # 调试
            # flag = regint(0)
            # target = abs(tar_elems1.reveal()[:])
            # target1 = abs(sin.reveal())
            # target2 = abs(cos.reveal())
            # for ii in range(n_elems):
            #     flag = (target[ii] > 0.05) + flag
            # # for ii in range(n_elems):
            # #     flag = (target1[ii] > 1.05) + flag
            # # for ii in range(n_elems):
            # #     flag = (target2[ii] > 1.05) + flag
            # @if_(flag * (times > 0))
            # def _():
            #     times.iadd(-1)
            #     print_ln("abs(tar_diff) count=%s k=%s: %s", count, k, abs(tar_diff).reveal())
            #     print_ln("sqrt count=%s k=%s: %s", count, k, (mpc_math.sqrt(4 * tar_elems1 * tar_elems1 + tar_diff * tar_diff)).reveal())
            #     print_ln("cos2θ count=%s k=%s: %s", count, k, cos2θ.reveal())
            #     print_ln("cos2 count=%s k=%s: %s", count, k, cos2.reveal())
            #     print_ln("cos count=%s k=%s: %s", count, k, cos.reveal())
            #     print_ln("sin count=%s k=%s: %s", count, k, sin.reveal())
            #     print_ln("A %s:", count)
            #     @for_range_opt(A.shape[0])
            #     def _(i):
            #         print_ln("%s",A[i].reveal())

            # 拼接上去
            A.assign(J.transpose().dot(A).dot(J))
            Q.assign(J.transpose().dot(Q))

        """ check for convergence"""
        sum_col = sfix.Array(n)
        # @for_range(n)
        # def _(i):
        for i in range(n): #******************
            sum_col[i] = array_sum(abs(A[i][:]))
        sum_matr = array_sum(sum_col)
        sum_diag = array_sum(abs(A.diag()))
        sum_offdiag = sum_matr - sum_diag
        print_ln("sum_offdiag: %s", sum_offdiag.reveal())
        @if_((sum_offdiag < (n * (n - 1) * tole)).reveal())
        def _():
            aa = 0
            break_loop()
            
    # print_ln("A:")
    # @for_range_opt(A.shape[0])
    # def _(i):
    #     print_ln("%s",A[i].reveal())

    print_ln("jacobi迭代次数: %s", count)
    eigenvals = sfix.Array(n)
    eigenvals.assign(A.diag())
    return eigenvals, Q

def test_jacobi():
    # A = np.array([
    # [0.700376, -0.477516, 0.172689, -0.397288, 0.0615101, 0.36293, -0.422701],
    # [-0.477516, 0.677556, -0.363702, 0.202558, -0.0365959, -0.546029, 0.543729],
    # [0.172689, -0.363702, 0.580317, -0.294371, 0.0781766, -0.341074, 0.167964],
    # [-0.397288, 0.202558, -0.294371, 0.565347, -0.0214278, 0.420273, -0.475092],
    # [0.0615101, -0.0365959, 0.0781766, -0.0214278, -0.0775892, -0.0166588, 0.0125851],
    # [0.36293, -0.546029, -0.341074, 0.420273, -0.0166588, 0.709948, -0.58939],
    # [-0.422701, 0.543729, 0.167964, -0.475092, 0.0125851, -0.58939, 0.762905]
    # ])
    n = 250
    np.random.seed(10)
    A = np.random.uniform(0, 1.6, size=(n, n))
    A = (A + A.T) / 2

    np.set_printoptions(linewidth=150)
    sfix.set_precision(30)
    sA = sfix.input_tensor_via(0, A)
    eigenvals, eigenvecs = jacobi(sA)
    # print_ln("eigenvals:\n%s", sort(eigenvals).reveal())
    # print_ln("eigenvecs:")
    # @for_range_opt(eigenvecs.shape[0])
    # def _(i):
    #     print_ln("%s",eigenvecs[i].reveal())

    
    # eigvals_np, eigvecs_np = np.linalg.eig(A)
    # print_ln("eigvals_np:%s\neigenvecs_np:%s", sorted(eigvals_np), eigvecs_np)

# program.use_trunc_pr = True # type: ignore
# test_jacobi()