# import jax.numpy as jnp
# import jax
# import numpy as np

# MAX_ITERATIONS = 5  # Generally, convergence is achieved within five iterations.

# def compute_elements(X, k, l, num_samples):
#     """Compute the rotation angle (cosine and sine) for the element X[k][l] in a matrix.

#     This function calculates the cosine and sine values required for a Jacobi rotation to eliminate the off-diagonal element X[k][l].

#     Args:
#         X: The input matrix for which the rotation angle is computed.
#         k: The row index of the target element.
#         l: The column index of the target element.
#         num_samples: The size of the matrix (number of rows/columns).

#     Returns:
#         cos: The cosine of the rotation angle.
#         sin: The sine of the rotation angle.
#     """
#     tar_elements = X[k][l]
#     tar_diff = X[k][k] - X[l][l]

#     cos_2theta = jnp.reciprocal(
#         jnp.sqrt(1 + 4 * jnp.square(tar_elements * jnp.reciprocal(tar_diff)))
#     )
#     cos2 = 0.5 + 0.5 * cos_2theta
#     sin2 = 0.5 - 0.5 * cos_2theta
#     flag_zero = jnp.equal(tar_elements, 0)
#     cos = jnp.sqrt(cos2) * (1 - flag_zero) + flag_zero
#     sin = (
#         (jnp.where(jnp.logical_and(tar_elements == 0, tar_diff == 0), 0, 1))
#         * jnp.sqrt(sin2)
#         * ((jnp.greater(tar_elements * tar_diff, 0)) * 2 - 1)
#     )

#     return cos, sin


# def rotation_matrix(X, k, l, num_samples):
#     """
#     Compute the Jacobi rotation matrix for eliminating the off-diagonal element X[k][l].

#     This function constructs a Jacobi rotation matrix `J`, which is used in the Jacobi eigenvalue algorithm
#     to zero out the off-diagonal element at position (k, l). The rotation is determined by computing the
#     cosine and sine values using the `compute_elements` function.

#     Args:
#         X: The input matrix for which the rotation matrix is computed.
#         k: The row index of the target off-diagonal element.
#         l: The column index of the target off-diagonal element.
#         num_samples: The size of the matrix (number of rows/columns).

#     Returns:
#         J: The Jacobi rotation matrix of size (num_samples, num_samples).
#     """
#     J = jnp.eye(num_samples)
#     k_values = jnp.array(k)
#     l_values = jnp.array(l)

#     # Parallelize using vmap
#     cos_values, sin_values = jax.vmap(compute_elements, in_axes=(None, 0, 0, None))(
#         X, k_values, l_values, num_samples
#     )

#     J = J.at[k, k].set(cos_values)
#     J = J.at[l, l].set(cos_values)
#     J = J.at[k, l].set(-sin_values)
#     J = J.at[l, k].set(sin_values)

#     return J


# # Ref:
# # https://arxiv.org/abs/2105.07612
# def Jacobi(X, num_samples):
#     """
#     Perform Jacobi eigenvalue decomposition on matrix X.

#     This function applies the Jacobi method to iteratively diagonalize the input matrix X.
#     The method rotates elements in the lower triangular part to eliminate off-diagonal elements.
#     It uses parallelized rotations to improve efficiency.

#     Args:
#         X: The input symmetric matrix (num_samples x num_samples) to be diagonalized.
#         num_samples: The size of the matrix (number of rows/columns).

#     Returns:
#         X: The diagonalized matrix (eigenvalues on the diagonal).
#         Q: The accumulated orthogonal transformation matrix (eigenvectors).
#     """
#     Q = jnp.eye(num_samples)
#     k = 0
#     while k < MAX_ITERATIONS:
#         # For each iteration, it is necessary to rotate all elements in the lower triangular part.
#         # To reduce the number of rounds, elements with non-repeating indices should be rotated in parallel as much as possible.
#         for i in range(1, num_samples + (num_samples - 1) // 2):
#             if i < num_samples:
#                 l_0 = i
#                 r_0 = 0
#             else:
#                 l_0 = num_samples - 1
#                 r_0 = i - l_0

#             n = (l_0 - r_0 - 1) // 2 + 1

#             j_indices = jnp.arange(n)
#             l = l_0 - j_indices
#             r = r_0 + j_indices

#             if i < num_samples // 2:
#                 l_1 = num_samples - 1 - r_0
#                 r_1 = num_samples - 1 - l_0
#                 n = (l_1 - r_1 - 1) // 2 + 1
#                 j_indices = jnp.arange(n)
#                 l_1 = l_1 - j_indices
#                 r_1 = r_1 + j_indices
#                 l = jnp.concatenate([l, l_1])
#                 r = jnp.concatenate([r, r_1])

#             # Calculate rotation matrix
#             J = rotation_matrix(X, l, r, num_samples)
#             # Update X and Q with rotation matrix
#             X = jnp.dot(J.T, jnp.dot(X, J))
#             Q = jnp.dot(J.T, Q)
#             # print("J: \n", J)
#             # print("Q: \n", Q)
#         k = k + 1

#     return X, Q



# num_samples = 7
# n_components = 3
# X = jnp.array([
#     [0.700376, -0.477516, 0.172689, -0.397288, 0.0615101, 0.36293, -0.422701],
#     [-0.477516, 0.677556, -0.363702, 0.202558, -0.0365959, -0.546029, 0.543729],
#     [0.172689, -0.363702, 0.580317, -0.294371, 0.0781766, -0.341074, 0.167964],
#     [-0.397288, 0.202558, -0.294371, 0.565347, -0.0214278, 0.420273, -0.475092],
#     [0.0615101, -0.0365959, 0.0781766, -0.0214278, -0.0775892, -0.0166588, 0.0125851],
#     [0.36293, -0.546029, -0.341074, 0.420273, -0.0166588, 0.709948, -0.58939],
#     [-0.422701, 0.543729, 0.167964, -0.475092, 0.0125851, -0.58939, 0.762905]
#     ])
# # print("X: \n", X)
# values, vectors = Jacobi(X, num_samples)
# values = jnp.diag(values)
# values = jnp.array(values)
# np.set_printoptions(linewidth=150)
# print("values: \n", values)
# print("vectors: \n", vectors)

# # Retrieve the largest n_components values and their corresponding vectors.
# # Sort each column of vectors according to values.
# vectors = vectors.T
# Index_value = jnp.argsort(values)
# values = values[Index_value]
# vectors = vectors[:, Index_value]
# # for i in range(num_samples):
# #     per_vectors = si.perm(vectors[i], Index_value)
# #     for j in range(num_samples):
# #         vectors = vectors.at[i, j].set(per_vectors[j])
# print("topk后：values: \n", values)
# print("topk后：vectors: \n", vectors)

# vectors = vectors[:, num_samples - n_components : num_samples]
# values = values[num_samples - n_components : num_samples]
# values = jnp.sqrt(values)
# values = jnp.diag(values)
# print("33：values: \n", values)
# print("33：vectors: \n", vectors)

# embedding_ = jnp.dot(vectors, values)
# embedding_ = embedding_[:, ::-1]
# print("embedding_: \n", embedding_)

import numpy as np
from sklearn.linear_model import LinearRegression

# 特征数据
n_train = np.array([1, 2, 3, 4, 5, 6]) * 10**5
# 目标数据
time_train = np.array([5.327068914, 10.67807416, 16.627721535, 21.869505738, 26.785295133, 33.103417313])

# Reshape data for sklearn
n_train = n_train.reshape(-1, 1)

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(n_train, time_train)

# 要预测的数据
n_test = np.array([7, 8, 9, 10]) * 10**5
n_test = n_test.reshape(-1, 1)

# 预测
predicted_time = model.predict(n_test)

# 输出预测结果
print(predicted_time)