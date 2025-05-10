# from Compiler.types import sint, sfix
# from Compiler.library import *
# T = 999


# def test_print_vector_secret(V):
#     [print_str("%s ", v.reveal()) for v in V]
#     print_ln(" ")


# def ineq(a, b):
#     return a < b


# def vector_permutation(v, factor):
#     return v


# def matrix_permutation(u, factor):
#     return u


# def obtain_random_factor(n):
#     return 0


# def exchange_elements(c, a, b):
#     aux_a = ternary_operator(c, b, a)
#     aux_b = ternary_operator(c, a, b)
#     return aux_a, aux_b


# def exchange_row_matrix(i, j, u):

#     for h in range(len(u)):
#         c = h == j
#         for k in range(len(u[0])):
#             u[i][k], u[h][k] = exchange_elements(c, u[i][k], u[h][k])
#     return u


# def exchange_vector(i, j, v):
#     for h in range(len(v)):
#         c = h == j
#         v[i], v[h] = exchange_elements(c, v[i], v[h])
#     return v


# def ternary_operator(c, if_true, if_false):
#     return c * (if_true - if_false) + if_false


# def dijkstra_optimized(weights, source):
#     ''' sfix distance and lighter compilation '''
#     n = len(weights)
#     distance = sfix.Array(n)
#     distance.assign_all(sfix(T))
#     # alpha = sfix.Array(n)
#     # alpha.assign_all(sfix(T))
#     # vertex_id = sfix.Array(n)
#     # @for_range(n)
#     # def _(i):
#     #     vertex_id[i] = sfix(i)
#     distance[source] = sfix(0)

#     # input shuffle
#     perm = sint.get_secure_shuffle(n)
#     weights.secure_permute(perm)
#     weights.transpose()
#     weights.secure_permute(perm)
#     weights.transpose()
    
#     distance.secure_permute(perm)
#     # vertex_id.secure_permute(perm)

#     p_weights = weights
#     # p_vertex_id = vertex_id

#     temp = sfix.Array(n)

#     @for_range(n)
#     def _(i):
#     # for i in range(n):
#         d_prime = sfix(T)
#         v = sint(0)
#         @for_range(n-1, i-1, -1)
#         def _(j):
#         # for j in range(n-1, i-1, -1):
#             c = ineq(distance[j], d_prime)
#             v.update(ternary_operator(c, j, v))
#             d_prime.update(ternary_operator(c, distance[j], d_prime))
#         v_open = None
#         if isinstance(v, sint):            
#             v_open = v.reveal()
#             # exchange_row_matri
#             temp.assign(p_weights[i])
#             p_weights[i] = p_weights[v_open]
#             p_weights[v_open] = temp
#             # exchange_vector
#             distance[i], distance[v_open] = distance[v_open], distance[i]
#             # p_vertex_id[i], p_vertex_id[v_open] = p_vertex_id[v_open], p_vertex_id[i]
#         else: 
#             v_open = v
#             exchange_row_matrix(i, v_open, p_weights)
#             exchange_vector(i, v_open, distance)
#             # exchange_vector(i, v_open, p_vertex_id)

#         @for_range_opt(i+1, n)
#         def _(j):
#         # for j in range(i+1, n):
#             a = distance[i] + p_weights[i][j]
#             c = ineq(a,  distance[j])
#             distance[j] = ternary_operator(c, a, distance[j])
#             # alpha[j] = ternary_operator(c, p_vertex_id[i], alpha[j])
#     return distance






# from Compiler.types import sint, sfix
# from Compiler.library import *
# T = 999


# def test_print_vector_secret(V):
#     [print_str("%s ", v.reveal()) for v in V]
#     print_ln(" ")


# def ineq(a, b):
#     return a < b


# def vector_permutation(v, factor):
#     return v


# def matrix_permutation(u, factor):
#     return u


# def obtain_random_factor(n):
#     return 0


# def exchange_elements(c, a, b):
#     aux_a = ternary_operator(c, b, a)
#     aux_b = ternary_operator(c, a, b)
#     return aux_a, aux_b


# def exchange_row_matrix(i, j, u):

#     for h in range(len(u)):
#         c = h == j
#         for k in range(len(u[0])):
#             u[i][k], u[h][k] = exchange_elements(c, u[i][k], u[h][k])
#     return u


# def exchange_vector(i, j, v):
#     for h in range(len(v)):
#         c = h == j
#         v[i], v[h] = exchange_elements(c, v[i], v[h])
#     return v


# def ternary_operator(c, if_true, if_false):
#     return c * (if_true - if_false) + if_false


# def dijkstra_optimized(weights, source):
#     n = len(weights)
#     # distance = [T] * n
#     # alpha = [T] * n
#     distance = sint.Array(n)
#     distance.assign_all(sint(T))
#     alpha = sint.Array(n)
#     alpha.assign_all(sint(T))
#     # vertex_id = [i for i in range(n)]
#     vertex_id = sint.Array(n)
#     @for_range(n)
#     def _(i):
#         vertex_id[i] = sint(i)
#     distance[source] = sint(0)

#     # 这里把洗牌省略了
#     factor = obtain_random_factor(0)
#     p_weights = matrix_permutation(weights, factor)
#     p_vertex_id = vector_permutation(vertex_id, factor)

#     temp = sint.Array(n)

#     @for_range(n)
#     def _(i):
#     # for i in range(n):
#         d_prime = sint(T)
#         v = sint(0)
#         @for_range(n-1, i-1, -1)
#         def _(j):
#         # for j in range(n-1, i-1, -1):
#             c = ineq(distance[j], d_prime)
#             v.update(ternary_operator(c, j, v))
#             print_ln("c: %s", c.reveal())
#             print_ln("v_open: %s", v.reveal())
#             d_prime.update(ternary_operator(c, distance[j], d_prime))
#         # v_open = None
#         if isinstance(v, sint):            
#             v_open = v.reveal()
#             # exchange_row_matri
#             # 为实现交换功能，为什么？
#             temp.assign(p_weights[i])
#             p_weights[i] = p_weights[v_open]
#             p_weights[v_open] = temp
#             # print_ln("temp: %s", temp.reveal())
#             print_ln("p_weights: %s", p_weights.reveal())
#             # print_ln("p_weights[v_open]: %s", p_weights[v_open].reveal())
#             # exchange_vector
#             distance[i], distance[v_open] = distance[v_open], distance[i]
#             p_vertex_id[i], p_vertex_id[v_open] = p_vertex_id[v_open], p_vertex_id[i]
#         else: 
#             v_open = v
#             exchange_row_matrix(i, v_open, p_weights)
#             exchange_vector(i, v_open, distance)
#             exchange_vector(i, v_open, p_vertex_id)

#         @for_range_opt(i+1, n)
#         def _(j):
#         # for j in range(i+1, n):
#             a = distance[i] + p_weights[i][j]
#             c = ineq(a,  distance[j])
#             distance[j] = ternary_operator(c, a, distance[j])
#             alpha[j] = ternary_operator(c, p_vertex_id[i], alpha[j])
#     return distance






















from Compiler.types import sint, sfix
from Compiler.library import if_then, end_if, print_ln, print_str
T = 999


def test_print_vector_secret(V):
    [print_str("%s ", v.reveal()) for v in V]
    print_ln(" ")


def ineq(a, b):
    return a < b


def vector_permutation(v, factor):
    return v


def matrix_permutation(u, factor):
    return u


def obtain_random_factor(n):
    return 0


def exchange_elements(c, a, b):
    aux_a = ternary_operator(c, b, a)
    aux_b = ternary_operator(c, a, b)
    return aux_a, aux_b


def exchange_row_matrix(i, j, u):

    for h in range(len(u)):
        c = h == j
        for k in range(len(u[0])):
            u[i][k], u[h][k] = exchange_elements(c, u[i][k], u[h][k])
    return u


def exchange_vector(i, j, v):
    for h in range(len(v)):
        c = h == j
        v[i], v[h] = exchange_elements(c, v[i], v[h])
    return v


def ternary_operator(c, if_true, if_false):
    return c * (if_true - if_false) + if_false


def dijkstra_optimized(weights, source):
    ''' sfix distance '''
    n = len(weights)
    distance = [T] * n
    alpha = [T] * n
    vertex_id = [i for i in range(n)]
    distance[source] = sfix(0)

    factor = obtain_random_factor(0)
    p_weights = matrix_permutation(weights, factor)
    p_vertex_id = vector_permutation(vertex_id, factor)

    for i in range(n):
        d_prime = T
        v = sint(0)
        for j in range(n-1, i-1, -1):
            c = ineq(distance[j], d_prime)
            v = ternary_operator(c, j, v)
            d_prime = ternary_operator(c, distance[j], d_prime)
        v_open = None
        if isinstance(v, sint):            
            v_open = v.reveal()
        else: 
            v_open = v
        exchange_row_matrix(i, v_open, p_weights)
        exchange_vector(i, v_open, distance)
        exchange_vector(i, v_open, p_vertex_id)

        for j in range(i+1, n):
            a = distance[i] + p_weights[i][j]
            c = ineq(a,  distance[j])
            distance[j] = ternary_operator(c, a, distance[j])
            alpha[j] = ternary_operator(c, p_vertex_id[i], alpha[j])
    return distance

# def dijkstra_optimized(weights, source):
#     n = len(weights)
#     distance = [T] * n
#     alpha = [T] * n
#     vertex_id = [i for i in range(n)]
#     distance[source] = sint(0)

#     factor = obtain_random_factor(0)
#     p_weights = matrix_permutation(weights, factor)
#     p_vertex_id = vector_permutation(vertex_id, factor)

#     for i in range(n):
#         d_prime = sint(T)
#         v = sint(0)
#         for j in range(n-1, i-1, -1):
#             c = ineq(distance[j], d_prime)
#             v = ternary_operator(c, j, v)
#             # print_ln("c: %s", (distance[j] < d_prime).reveal())
#             print_ln("v_open: %s", v.reveal())
#             d_prime = ternary_operator(c, distance[j], d_prime)
#         v_open = None
#         if isinstance(v, sint):            
#             v_open = v.reveal()
#         else: 
#             v_open = v
#         exchange_row_matrix(i, v_open, p_weights)
#         exchange_vector(i, v_open, distance)
#         exchange_vector(i, v_open, p_vertex_id)

#         # print_ln("p_weights[i]: %s", p_weights[i].reveal())
#         # print_ln("p_weights[v_open]: %s", p_weights[v_open].reveal())
#         print_ln("p_weights: %s", p_weights.reveal())

#         for j in range(i+1, n):
#             a = distance[i] + p_weights[i][j]
#             c = ineq(a,  distance[j])
#             distance[j] = ternary_operator(c, a, distance[j])
#             alpha[j] = ternary_operator(c, p_vertex_id[i], alpha[j])
#     return distance