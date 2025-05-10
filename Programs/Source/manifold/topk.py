from Compiler.library import *

def patition1(arr, pi, low, high):
    """
    通信量少，但不完全并行
    """

    # print_ln("arr0:")
    # arr.print_reveal_nested("\n")
    # print_ln("pi0:")
    # pi.print_reveal_nested("\n")


    i = MemValue(low-1)  # 使用 MemValue 来处理可变变量
    c = Array(len(arr), sint)

    # n = 10 , buget = 3000, rounds = 85
    # n = 100, buget = 12000, rounds = 364
    # n = 100, buget = 52000, rounds = 196
    # n = 10000, buget = 5200000, rounds = 259
    @for_range_opt(high - low) 
    def _(j):
        c[low  + j] = arr[low  + j] < arr[high]
    c = c.reveal()

    @for_range_opt(high - low)
    def _(j):
        @if_(c[low  + j])
        def _():
            i.write(i.read()+ 1)
            arr[i.read()], arr[j + low] = arr[j + low], arr[i.read()]
            # process permutation
            pi[i.read()], pi[j + low] = pi[j + low], pi[i.read()]
    p = i.iadd(1)
    arr[p], arr[high] = arr[high], arr[p]
    # process permutation
    pi[p], pi[high] = pi[high], pi[p]
    return p


def patition2(arr, low, high):
    """
    通信量多，但完全并行
    """
    i = MemValue(low-1)  # 使用 MemValue 来处理可变变量  i = MemValue(low-1)
    cc = Array(len(arr), sint)
    c = Array(len(arr), cint)
    # c = Array(len(arr), sint)
    # idx = Array(len(arr), cint)

    # arr_pivot：arr[high] 的序列
    arr_pivot = Array(len(arr), sfix)
    arr_pivot.assign_all(arr[high])
    # @for_range(high - low)
    # def _(j):
    #     # idx[j] = cint(low + j)
    #     arr_pivot[low + j] = arr[high]
    # # idx.print_reveal_nested(end='\n')
    # arr_pivot.print_reveal_nested(end='\n')

    # print_ln('part: %s', arr.get_slice_vector(idx).reveal())

    # cc = arr[low : high] >= arr_pivot[low : high]
    cc = arr[:] >= arr_pivot[:]
    c.assign(cc.reveal())
    # print_ln('c: %s', c)

    # @for_range_opt(high - low)
    # def _(j):
    #     c[low  + j] = arr[low  + j] >= arr[high]
    # c = c.reveal()

    @for_range_opt(high - low)
    def _(j):
        @if_(c[low  + j])
        def _():
            i.write(i.read()+ 1)
            temp = arr[i.read()]
            arr[i.read()] = arr[j + low]
            arr[j + low] = temp
    p = i.iadd(1)
    # print_ln('p: %s', p)
    temp = arr[p]
    arr[p] = arr[high]
    arr[high] = temp
    return p

def topk(arr, k):
    # pi = sint.input_tensor_via(0, list(range(len(arr))))
    pi = sint.Array(len(arr))
    for i in range(len(arr)):
    # @for_range_opt(len(arr))
    # def _(i):
        pi[i] = sint(i)
    
    handle = sint.get_secure_shuffle(len(arr))
    pi.secure_permute(handle)
    arr.secure_permute(handle)

    # delshuffle(handle)

    # print_ln("pi_in: %s", pi.reveal())
    low = MemValue(0)
    high = MemValue(len(arr)-1)
    i = MemValue(0)
    @do_while
    def _():    
        i.iadd(1)
        p = patition1(arr, pi, low, high)

        # print_ln("arr0:")
        # arr.print_reveal_nested("\n")
        # print_ln("pi0:")
        # pi.print_reveal_nested("\n")

        @if_e(p < k)
        def _():
            low.write(p.read() + 1)
        @else_
        def _():    
            high.write(p.read() - 1)
        # return i.read() != 1
        # return (p.read() != (k+1)) & (p.read() != k)

        # print_ln('p: %s', p.read())
        # print_ln('arr_parted:')
        # arr.print_reveal_nested(end='\n')

        # print_ln('p: %s', p.reveal())
        return (p.read() != (k-1)) & (p.read() != k)
    # print_ln('num of patitions: %s', i.reveal())
    return pi


def topk_baseline(arr, k):
    # pi = sint.input_tensor_via(0, list(range(len(arr))))
    pi = sint.Array(len(arr))
    for i in range(len(arr)):
    # @for_range_opt(len(arr))
    # def _(i):
        pi[i] = sint(i)
    
    handle = sint.get_secure_shuffle(len(arr))
    pi.secure_permute(handle)
    arr.secure_permute(handle)
    low = MemValue(0)
    high = MemValue(len(arr)-1)
    i = MemValue(0)
    @do_while
    def _():    
        i.iadd(1)
        p = patition1(arr, pi, low, high)
        @if_e(p < k)
        def _():
            low.write(p.read() + 1)
        @else_
        def _():    
            high.write(p.read() - 1)
        # print_ln('p: %s', p.reveal())
        return p.read() != (k-1)
    print_ln('num of patitions: %s', i.reveal())
    return pi
