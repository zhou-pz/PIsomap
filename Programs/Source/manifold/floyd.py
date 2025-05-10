from Compiler.library import *
import numpy as np
from scipy.sparse.csgraph import floyd_warshall # type: ignore
# program.use_split(3)

def delete_row_k(M, k):
    n = len(M)
    out = Matrix(n-1, M.shape[1], sfix)
    for i in range (n-1):
        if(i < k):
            out[i][:] = M[i][:]
        else:
            out[i][:] = M[i+1][:]
    return out

def delete_col_k(M, k):
    M = M.transpose()
    return delete_row_k(M, k).transpose()

def delete_row_k_1(M, k):
    n = len(M)
    out = Matrix(n-1, M.shape[1], sfix)
    for i in range (n-1):
        @if_e(i < k)
        def _():
            out[i][:] = M[i][:]
        @else_
        def _():
            out[i][:] = M[i+1][:]
    return out

def delete_col_k_1(M, k):
    M = M.transpose()
    return delete_row_k_1(M, k).transpose()

def floyd_opt(
    dist
):
    n = len(dist)
    indices = np.column_stack(np.triu_indices(n-1, k=1)).tolist()
    for k in range(n):
    # for k in range(2):
        # 打包计算
        # 删掉第k行和第k列
        # batch = jnp.delete(batch, k, axis=0)
        # print_ln('batch: ')
        # batch.print_reveal_nested(end='\n')
        # print_ln('dist f0: k = %s', k)
        # dist.print_reveal_nested(end='\n')
        batch = delete_col_k(dist, k)
        row_k_without_dkk = batch[k]
        # print_ln('row_k_without_dkk=%s', row_k_without_dkk.reveal())
        batch = delete_row_k(batch, k)
        # print_ln('row_k_without_dkk: ')
        # row_k_without_dkk.print_reveal_nested(end='\n')

        # 设置d_i,k和d_k,j的值
        dist_ik = batch.same_shape()  
        dist_kj = batch.same_shape()

        for i in range(n-1):
            dist_kj[i] = row_k_without_dkk
        dist_ik = dist_kj.transpose()
        # print_ln('dist_ik: ')
        # dist_ik.print_reveal_nested(end='\n')
        # print_ln('dist_kj: ')
        # dist_kj.print_reveal_nested(end='\n')

        # # 把上三角拿出来算
        # batch_2_upper_triangle = batch[indices]
        # dist_ik_upper_triangle = dist_ik[indices]
        # dist_kj_upper_triangle = dist_kj[indices]
        npoint = len(indices)
        batch_2_upper_triangle = Array(npoint, sfix)
        dist_ik_upper_triangle = Array(npoint, sfix)
        dist_kj_upper_triangle = Array(npoint, sfix)

        for i in range(npoint):
            batch_2_upper_triangle[i] = batch[indices[i][0]][indices[i][1]]
            dist_ik_upper_triangle[i] = dist_ik[indices[i][0]][indices[i][1]]
            dist_kj_upper_triangle[i] = dist_kj[indices[i][0]][indices[i][1]]
        # print_ln('batch: ')
        # batch.print_reveal_nested(end='\n')
        # print_ln('batch_2_upper_triangle: ')
        # batch_2_upper_triangle.print_reveal_nested(end='\n')

        # batch_2_upper_triangle = jnp.minimum(batch_2_upper_triangle, dist_ik_upper_triangle + dist_kj_upper_triangle)
        dist_ik_upper_triangle = dist_ik_upper_triangle + dist_kj_upper_triangle
        batch_2_upper_triangle = dist_ik_upper_triangle + (batch_2_upper_triangle < dist_ik_upper_triangle) * (batch_2_upper_triangle - dist_ik_upper_triangle)
        # print_ln('batch_2_upper_triangle f000: k = %s\n %s', k, batch_2_upper_triangle.reveal())

        # print_ln('indices f0: k = %s: %s', k, indices)

        # 把上三角放回去
        batch = batch.same_shape()
        # print_ln('batch pre: k = %s', k)
        # batch.print_reveal_nested(end='\n')
        # batch = batch.at[indices].set(batch_2_upper_triangle)
        for i in range(npoint):
            # print_ln('batch_2_upper_triangle[i] f000: k = %s\n %s', k, batch_2_upper_triangle[i].reveal())
            # print_ln('batch[.][.] pre f000: k = %s\n %s', k, batch[indices[i][0]][indices[i][1]].reveal())
            batch[indices[i][0]][indices[i][1]] = batch_2_upper_triangle[i]
            # print_ln('batch[.][.] pos f000: k = %s\n %s', k, batch[indices[i][0]][indices[i][1]].reveal())

        batch += batch.transpose()
        # print_ln('batch pos: k = %s', k)
        # batch.print_reveal_nested(end='\n')

        # batch = jnp.insert(batch, k, col_k_without_dkk, axis=1)      # 把更新的值放回原位置
        # batch = jnp.insert(batch, k, dist[k], axis=0)
        def insert_row_k(batch, k, row_k_without_dkk):
            n = len(batch)
            temp1 = Matrix(n+1, batch.shape[1], sfix)
            for i in range(n+1):
                if(i<k):
                    temp1[i] = batch[i]
                elif(i>k):
                    temp1[i] = batch[i-1]
                else:
                    temp1[i] = row_k_without_dkk
            return temp1
        
        def insert_col_k(M, k):
            M = M.transpose()
            return insert_row_k(M, k, dist.get_column(k)).transpose()

        temp1 = insert_row_k(batch, k, row_k_without_dkk)
        # print_ln('temp1 f0: k = %s', k)
        # temp1.print_reveal_nested(end='\n')

        # print_ln('batch: ')
        # batch.print_reveal_nested(end='\n')
        # print_ln('row_k_without_dkk: ')
        # row_k_without_dkk.print_reveal_nested(end='\n')
        # print_ln('temp1: ')
        # temp1.print_reveal_nested(end='\n')


        dist = insert_col_k(temp1, k)
    return dist

def floyd_opt_1(
    dist 
):
    n = len(dist)
    indices = np.column_stack(np.triu_indices(n-1, k=1)).tolist()
    batch_temp = sfix.Matrix(n-1, n-1)
    @for_range(n)
    # @for_range(2)
    def _(k):
        nonlocal dist
        nonlocal batch_temp
        # print_ln('dist f1: k = %s', k)
        # dist.print_reveal_nested(end='\n')
        batch = delete_col_k_1(dist, k)
        row_k_without_dkk = batch[k]
        batch = delete_row_k_1(batch, k)
        # print_ln('row_k_without_dkk: ')
        # row_k_without_dkk.print_reveal_nested(end='\n')

        dist_ik = batch.same_shape()  
        dist_kj = batch.same_shape()

        for i in range(n-1):
            dist_kj[i] = row_k_without_dkk
        dist_ik = dist_kj.transpose()

        npoint = len(indices)
        batch_2_upper_triangle = Array(npoint, sfix)
        dist_ik_upper_triangle = Array(npoint, sfix)
        dist_kj_upper_triangle = Array(npoint, sfix)

        for i in range(npoint):
            batch_2_upper_triangle[i] = batch[indices[i][0]][indices[i][1]]
            dist_ik_upper_triangle[i] = dist_ik[indices[i][0]][indices[i][1]]
            dist_kj_upper_triangle[i] = dist_kj[indices[i][0]][indices[i][1]]

        dist_ik_upper_triangle = dist_ik_upper_triangle + dist_kj_upper_triangle
        batch_2_upper_triangle = dist_ik_upper_triangle + (batch_2_upper_triangle < dist_ik_upper_triangle) * (batch_2_upper_triangle - dist_ik_upper_triangle)
        # print_ln('batch_2_upper_triangle f111: k = %s\n %s', k, batch_2_upper_triangle.reveal())

        # print_ln('indices: k = %s: %s', k, indices)

        batch.assign(batch_temp)
        # print_ln('batch pre: k = %s', k)
        # batch.print_reveal_nested(end='\n')
        for i in range(npoint):
            # print_ln('batch_2_upper_triangle[i] f111: k = %s\n %s', k, batch_2_upper_triangle[i].reveal())
            # print_ln('batch[.][.] pre f111: k = %s\n %s', k, batch[indices[i][0]][indices[i][1]].reveal())
            batch[indices[i][0]][indices[i][1]] = batch_2_upper_triangle[i]
            # print_ln('batch[.][.] pos f111: k = %s\n %s', k, batch[indices[i][0]][indices[i][1]].reveal())
        batch = batch + batch.transpose()
        # print_ln('batch pos: k = %s', k)
        # batch.print_reveal_nested(end='\n')

        def insert_row_k(batch, k, inrow):
            n = len(batch)
            temp = Matrix(n+1, batch.shape[1], sfix)
            @for_range_opt(n+1)
            def _(i):
            # for i in range(n+1):
                @if_e(i<k)
                def _():
                    temp[i] = batch[i]
                @else_
                def _():
                    @if_e(i>k)
                    def _():
                        temp[i] = batch[i-1]
                    @else_
                    def _():
                        temp[i] = inrow
            return temp
        
        def insert_col_k(M, k, incol):
            M = M.transpose()
            return insert_row_k(M, k, incol).transpose()

        temp1 = insert_row_k(batch, k, row_k_without_dkk)

        # print_ln('temp1 f1: k = %s', k)
        # temp1.print_reveal_nested(end='\n')
        # print_ln('row_k_without_dkk: ')
        # row_k_without_dkk.print_reveal_nested(end='\n')

        dist.assign(insert_col_k(temp1, k, dist.get_column(k)))

    return dist


def floyd_opt_2(
    dist
):
    n = len(dist)
    # 生成上三角索引
    Indices = np.column_stack(np.triu_indices(n, k=1)).tolist()
    for k in range(n):
        # 去除 i=k or j=k 的索引
        indices = [pair for pair in Indices if not (pair[0] == k or pair[1] == k)]
        for idx in indices:
            candidate = dist[idx[0]][k] + dist[k][idx[1]]
            # dist[idx[1]][idx[0]] = dist[idx[0]][idx[1]] = \
            #     candidate + (dist[idx[0]][idx[1]] < candidate) * (dist[idx[0]][idx[1]] - candidate)
            dist[idx[1]][idx[0]] = dist[idx[0]][idx[1]] = dist[idx[0]][idx[1]].min(candidate)
    return dist

def floyd_opt_3(
    dist
):
    n = len(dist)
    # 生成上三角索引
    Indices = np.column_stack(np.triu_indices(n, k=1)).tolist()
    @for_range(n)
    def _(k):
        # 去除 i=k or j=k 的索引
        indices = []  
        for pair in Indices:
            @if_((pair[0] != k) * (pair[1] != k))
            def _():
                indices.append(pair)

        for idx in indices:
            candidate = dist[idx[0]][k] + dist[k][idx[1]]
            dist[idx[1]][idx[0]] = dist[idx[0]][idx[1]] = dist[idx[0]][idx[1]].min(candidate)
    return dist
