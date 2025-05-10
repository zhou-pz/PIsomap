# import numpy as np

# def wilkinson_shift(A):
#     """计算 Wilkinson Shift"""
#     n = A.shape[0]
#     if n < 2:
#         return A[-1, -1]
#     a, b, c, d = A[-2, -2], A[-2, -1], A[-1, -2], A[-1, -1]
#     delta = (a - d) / 2.0
#     sign_delta = np.sign(delta) if delta != 0 else 1
#     shift = d - sign_delta * (b * b) / (abs(delta) + np.sqrt(delta**2 + b * c))
#     return shift

# def rayleigh_shift(A, v):
#     """计算 Rayleigh Quotient Shift"""
#     return (v.T @ A @ v) / (v.T @ v)

# def qr_shift_eigen(A, shift_type="rayleigh", tol=1e-8):
#     """
#     QR Shift 计算矩阵特征值和特征向量
#     :param A: 输入矩阵 (必须是方阵)
#     :param shift_type: 选择 shift 方法 ("wilkinson" or "rayleigh")
#     :param tol: 误差容忍度
#     :param max_iter: 最大迭代次数
#     :return: 近似特征值和特征向量
#     """
#     n = A.shape[0]
#     A_k = A.copy()
#     Q_total = np.eye(n)
#     max_iter = min(10 * n ** 2, 100000)  # 设置最大迭代次数

#     for _ in range(max_iter):
#         if shift_type == "wilkinson":
#             mu = wilkinson_shift(A_k)
#         elif shift_type == "rayleigh":
#             v = np.random.rand(n, 1)
#             mu = rayleigh_shift(A_k, v)
#         else:
#             raise ValueError("Unsupported shift type!")

#         Q, R = np.linalg.qr(A_k - mu * np.eye(n))
#         A_k = R @ Q + mu * np.eye(n)
#         Q_total = Q_total @ Q

#         if np.all(np.abs(A_k[np.tril_indices(n, -1)]) < tol):
#             break

#     eigenvalues = np.diag(A_k)
#     eigenvectors = Q_total
#     return eigenvalues, eigenvectors

# def lanczos(A, k):
#     """
#     Lanczos 算法：用于近似计算对称矩阵的前 k 大特征值
#     """
#     n = A.shape[0]
#     Q = np.zeros((n, k))
#     T = np.zeros((k, k))

#     q = np.random.rand(n)
#     q /= np.linalg.norm(q)  # 初始单位向量
#     Q[:, 0] = q

#     beta = 0
#     for j in range(k):
#         z = A @ q
#         alpha = np.dot(q, z)  # 计算 α_j
#         if j > 0:
#             z -= beta * Q[:, j - 1]
#         z -= alpha * q
#         beta = np.linalg.norm(z)

#         T[j, j] = alpha
#         if j < k - 1:
#             T[j, j + 1] = T[j + 1, j] = beta
#             q = z / beta
#             Q[:, j + 1] = q

#     eigenvalues, eigenvectors = np.linalg.eig(T)
#     return eigenvalues, Q @ eigenvectors  # 近似特征值和特征向量

# def verify_eigen(A, eigenvalues, eigenvectors, tol=1e-6):
#     """
#     验证特征值和特征向量的正确性
#     """
#     n = eigenvalues.shape[0]
#     errors = []
#     for i in range(n):
#         λ = eigenvalues[i]
#         v = eigenvectors[:, i].reshape(-1, 1)

#         # 计算误差 || A v - λ v ||
#         error = np.linalg.norm(A @ v - λ * v)
#         errors.append(error)
    
#     # 统计误差情况
#     max_error = max(errors)
#     print(f"Max Eigenvector Error: {max_error:.6e} {'✅' if max_error < tol else '❌'}")

# # 生成测试矩阵（随机对称矩阵）
# np.random.seed(42)
# n = 10  # 矩阵维度
# A = np.random.rand(n, n)
# A = (A + A.T)  # 生成对称矩阵，保证实特征值

# # 计算特征值和特征向量 (QR Shift)
# eigvals_w, eigvecs_w = qr_shift_eigen(A, shift_type="rayleigh")

# # 计算特征值和特征向量 (NumPy)
# eigvals_np, eigvecs_np = np.linalg.eig(A)

# # 输出并验证 QR Shift 计算结果
# # print("\nQR with Wilkinson Shift:")
# # print("Eigenvalues:\n", eigvals_w)
# print("\nVerification of Wilkinson Shift Results:")
# verify_eigen(A, eigvals_w, eigvecs_w)

# # **Lanczos 计算**
# k = 5  # 近似计算前 10 个最大特征值
# eigvals_lanczos, eigvecs_lanczos = lanczos(A, k)
# print("\nVerification of lanczos Results:")
# verify_eigen(A, eigvals_lanczos, eigvecs_lanczos)

# # 输出 NumPy 计算结果并验证
# print("\nVerification of NumPy Results:")
# verify_eigen(A, eigvals_np, eigvecs_np)



# eigvals_A, _ = np.linalg.eigh(A)  # A 的精确特征值
# eigvals_A_largest = np.sort(eigvals_A)[-k:]  # 取前 k 大的特征值
# print("Lanczos 近似特征值:", np.sort(eigvals_lanczos))
# print("A 的前 k 大特征值:", eigvals_A_largest)




import numpy as np
import sklearn.manifold

def householder_qr(A):
    """
    利用 Householder 反射计算矩阵 A 的 QR 分解
    返回正交矩阵 Q 和上三角矩阵 R，使得 A = Q @ R
    """
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)
    
    for k in range(n):
        # 取 A[k:, k] 作为当前列向量
        x = A[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:
            print("norm_x == 0")
            continue

        # 确定反射向量，避免数值不稳定性（选择符号使得 u 的第一个分量为正）
        sign = -1 if x[0] < 0 else 1
        u1 = x[0] + sign * norm_x
        v = x.copy()
        v[0] = u1
        # 归一化 v
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            print("norm_v == 0")
            continue
        v = v / norm_v
        
        # 构造 Householder 矩阵 H = I - 2 v v^T，作用在 A 的第 k 行及之后的部分
        H_k = np.eye(m - k) - 2.0 * np.outer(v, v)
        
        # 更新 A 的第 k 行及之后的部分：A[k:, k:] = H_k @ A[k:, k:]
        A[k:, k:] = H_k @ A[k:, k:]
        
        # 通过扩展 H_k 为全 m 维矩阵，更新 Q。注意：Q 乘上扩展后的 H_k
        H_full = np.eye(m)
        H_full[k:, k:] = H_k

        # print("Q", k, "\n", Q)
        # print("H_full", k, "\n", H_full)

        Q = Q @ H_full  # 最终 Q = H_full_0 * H_full_1 * ... * H_full_{n-1}

        # print("Q", k, "\n", Q)
    
    R = A
    return Q, R

def rayleigh_shift(A, v):
    """计算 Rayleigh 商作为移位参数"""
    return (v.T @ A @ v) / (v.T @ v)

def qr_shift_eigen(A, tol=1e-2):
    """
    用 Rayleigh 移位的 QR 算法计算矩阵特征值和特征向量
    :param A: 方阵
    :param tol: 收敛容差
    :return: 近似特征值（对角线）和特征向量矩阵
    """
    n = A.shape[0]
    A_k = A.copy()
    Q_total = np.eye(n)
    max_iter = min(10 * n ** 2, 100000)  # 最大迭代次数
    # flag = True
    count = 0

    for _ in range(max_iter):
        count += 1
        # 生成随机向量 v 用于 Rayleigh 商计算
        v = np.random.rand(n, 1)
        mu = rayleigh_shift(A_k, v)
        
        Q, R = np.linalg.qr(A_k - mu * np.eye(n))
        # # 自定义 QR 分解，计算 A_k - mu*I 的 QR 分解
        # Q, R = householder_qr(A_k - mu * np.eye(n))

        # # 检查原矩阵是否能被还原
        # A_reconstructed = Q @ R
        # if np.allclose(A_k - mu * np.eye(n), A_reconstructed, rtol=1e-10, atol=1e-10):
        #     flag = flag
        # else:
        #     flag = False

        # 更新 A_k 保持相似性：A_k+1 = R Q + mu I
        A_k = R @ Q + mu * np.eye(n)
        # 累积所有迭代中的正交变换
        Q_total = Q_total @ Q

        # 收敛判定：当 A_k 的下三角非对角部分均足够小时退出
        if np.all(np.abs(A_k[np.tril_indices(n, -1)]) < tol):
            break

    eigenvalues = np.diag(A_k)
    eigenvectors = Q_total
    print("迭代次数：", count)

    # print(flag)
    return eigenvalues, eigenvectors

def qr_shift_eigen2(A, tol=1e-2):   # tol不能设太高，导致过度迭代，反而影响特征精度
    """
    用 Rayleigh 移位的 QR 算法计算矩阵特征值和特征向量
    :param A: 方阵
    :param tol: 收敛容差
    :return: 近似特征值（对角线）和特征向量矩阵
    """
    n = A.shape[0]
    A_k = A.copy()
    Q_total = np.eye(n)
    max_iter = min(10 * n ** 2, 10000)  # 最大迭代次数
    flag = True
    count = 0

    # print("A_k", A_k)

    # 生成随机向量 v 用于 Rayleigh 商计算
    v = np.random.rand(n, 1)
    for _ in range(40):
        count += 1
        # mu = rayleigh_shift(A_k, v)
        # print("mu: ", mu)
        
        # Q, R = np.linalg.qr(A_k - mu * np.eye(n))
        # # 自定义 QR 分解，计算 A_k - mu*I 的 QR 分解
        Q, R = householder_qr(A_k)
        # print("Q: ", Q)
        # print("R: ", R)

        # # 检查原矩阵是否能被还原
        # A_reconstructed = Q @ R
        # if np.allclose(A_k - mu * np.eye(n), A_reconstructed, rtol=1e-10, atol=1e-10):
        #     flag = flag
        # else:
        #     flag = False

        # 更新 A_k 保持相似性：A_k+1 = R Q + mu I
        A_k = R @ Q
        # 累积所有迭代中的正交变换
        Q_total = Q_total @ Q

        # print("A_k", A_k)

        # 收敛判定：当 A_k 的下三角非对角部分均足够小时退出
        if np.all(np.abs(A_k[np.tril_indices(n, -1)]) < tol):
            break
        # flag = 1
        # for i in range(n - 1):
        #     if abs(A_k[i+1, i]) > (tol * (abs(A_k[i, i]) + abs(A_k[i+1, i+1]))):
        #         flag = 0
        # if flag:
        #     break


    print("A_k", sorted(np.diag(A_k)))

    print("迭代次数：", count)
    eigenvalues = np.diag(A_k)
    eigenvectors = Q_total

    print(flag)
    return eigenvalues, eigenvectors


def verify_eigen(A, eigenvalues, eigenvectors, tol=1e-2):
    """
    验证特征值和特征向量的正确性：计算每个 (λ, v) 对的残差 ||Av - λv||
    """
    n = eigenvalues.shape[0]
    errors = []
    for i in range(n):
        lam = eigenvalues[i]
        v = eigenvectors[:, i].reshape(-1, 1)
        error = np.linalg.norm(A @ v - lam * v)
        errors.append(error)
    max_error = max(errors)
    print(f"Max Eigenvector Error: {max_error:.6e} {'✅' if max_error < tol else '❌'}")

# 主程序
np.random.seed(4536)
n = 10  # 矩阵维度
A = np.random.rand(n, n)
A = A + A.T  # 构造对称矩阵，保证实特征值

# # 使用自定义的 QR Shift 算法计算特征值与特征向量
# eigvals_qr, eigvecs_qr = qr_shift_eigen(A)
# print("\nQR Shift 库:", eigvals_qr, eigvecs_qr)
# print("\nVerification of QR Shift Results:")
# verify_eigen(A.copy(), eigvals_qr, eigvecs_qr)

np.set_printoptions(suppress=True, precision=6)

eigvals_qr2, eigvecs_qr2 = qr_shift_eigen2(A)
# print("\nQR Shift 自定义:", eigvals_qr2, eigvecs_qr2)
print("\nVerification of QR Shift Results:")
verify_eigen(A.copy(), eigvals_qr2, eigvecs_qr2)

# # 利用 NumPy 自带的 np.linalg.eig 进行比较
# eigvals_np, eigvecs_np = np.linalg.eig(A)
# print("\nVerification of NumPy Results:")
# verify_eigen(A, eigvals_np, eigvecs_np)

