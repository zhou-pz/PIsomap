/*
 * SecureShuffle.hpp
 *
 */

#ifndef PROTOCOLS_SECURESHUFFLE_HPP_
#define PROTOCOLS_SECURESHUFFLE_HPP_

#include "SecureShuffle.h"
#include "Tools/Waksman.h"

#include <math.h>
#include <algorithm>

template<class T>
void ShuffleStore<T>::lock()
{
    store_lock.lock();
}

template<class T>
void ShuffleStore<T>::unlock()
{
    store_lock.unlock();
}

template<class T>
int ShuffleStore<T>::add()
{
    lock();
    int res = shuffles.size();
    shuffles.push_back({});
    unlock();
    return res;
}

template<class T>
typename ShuffleStore<T>::shuffle_type& ShuffleStore<T>::get(int handle)
{
    lock();
    auto& res = shuffles.at(handle);
    unlock();
    return res;
}

template<class T>
void ShuffleStore<T>::del(int handle)
{
    lock();
    shuffles.at(handle) = {};
    unlock();
}

template<class T>
SecureShuffle<T>::SecureShuffle(SubProcessor<T>& proc) :
        proc(proc), unit_size(0), n_shuffle(0), exact(false)
{
}

template<class T>
SecureShuffle<T>::SecureShuffle(vector<T>& a, size_t n, int unit_size,
        size_t output_base, size_t input_base, SubProcessor<T>& proc) :
        proc(proc), unit_size(unit_size), n_shuffle(0), exact(false)
{
    pre(a, n, input_base);

    for (auto i : proc.protocol.get_relevant_players())
        player_round(i);

    post(a, n, output_base);
}

template<class T>
void SecureShuffle<T>::apply(vector<T>& a, size_t n, int unit_size, size_t output_base,
        size_t input_base, shuffle_type& shuffle, bool reverse)
{
    this->unit_size = unit_size;

    pre(a, n, input_base);

    assert(shuffle.size() == proc.protocol.get_relevant_players().size());

    if (reverse)
        for (auto it = shuffle.end(); it > shuffle.begin(); it--)
        {
            this->config = *(it - 1);
            iter_waksman(reverse);
        }
    else
        for (auto& config : shuffle)
        {
            this->config = config;
            iter_waksman(reverse);
        }

    post(a, n, output_base);
}


template<class T>
void SecureShuffle<T>::applyMultiple(vector<T>& a, vector<size_t>& sizes, vector<size_t>& destinations, vector<size_t>& sources,
                                    vector<size_t>& unit_sizes, vector<size_t>& handles, vector<bool>& reverse, store_type& store) {
    vector<shuffle_type> shuffles;
    for (size_t &handle : handles)
        shuffles.push_back(store.get(handle));

    this->applyMultiple(a, sizes, destinations, sources, unit_sizes, shuffles, reverse);
}

template<class T>
void SecureShuffle<T>::applyMultiple(vector<T> &a, vector<size_t> &sizes, vector<size_t> &destinations,
    vector<size_t> &sources, vector<size_t> &unit_sizes, vector<shuffle_type> &shuffles, vector<bool> &reverse) {
    const auto n_shuffles = sizes.size();
    assert(sources.size() == n_shuffles);
    assert(destinations.size() == n_shuffles);
    assert(unit_sizes.size() == n_shuffles);
    assert(shuffles.size() == n_shuffles);
    assert(reverse.size() == n_shuffles);

    // SecurePermute works by making t players create and "secret-share" a permutation.
    // Then each permutation is applied in a pass. As long as one of these permutations was created by a honest party,
    // the resulting combined shuffle is hidden.
    const auto n_passes = proc.protocol.get_relevant_players().size();

    // Initialize the shuffles.
    vector<bool> isExact(n_shuffles, false);
    vector<vector<T>> toShuffle;
    int max_depth = 0;

    for (size_t currentShuffle = 0; currentShuffle < n_shuffles; currentShuffle++) {
        const size_t input_base = sources[currentShuffle];
        const size_t n = sizes[currentShuffle];
        const size_t unit_size = unit_sizes[currentShuffle];

        assert(shuffles[currentShuffle].size() == n_passes);
        assert(n % unit_size == 0);

        const size_t n_shuffle = n / unit_size;
        const size_t n_shuffle_pow2 = (1u << int(ceil(log2(n_shuffle))));
        const bool exact = (n_shuffle_pow2 == n_shuffle) or not T::malicious;

        if (log2(n_shuffle) > max_depth)
            max_depth = log2(n_shuffle);

        vector<T> tmp;
        if (exact)
        {
            tmp.resize(n_shuffle_pow2 * unit_size);
            for (size_t i = 0; i < n; i++)
                tmp[i] = a[input_base + i];
        }
        else
        {
            // Pad n_shuffle to n_shuffle_pow2. To reduce this back to n_shuffle after-the-fact, a flag bit is
            // added to every real entry.
            const size_t shuffle_unit_size = unit_size + 1;
            tmp.resize(shuffle_unit_size * n_shuffle_pow2);
            for (size_t i = 0; i < n_shuffle; i++)
            {
                for (size_t j = 0; j < unit_size; j++)
                    tmp[i * shuffle_unit_size + j] = a[input_base + i * unit_size + j];
                tmp[(i + 1) * shuffle_unit_size - 1] = T::constant(1, proc.P.my_num(), proc.MC.get_alphai());
            }
            for (size_t i = n_shuffle * shuffle_unit_size; i < tmp.size(); i++)
                tmp[i] = T::constant(0, proc.P.my_num(), proc.MC.get_alphai());
            unit_sizes[currentShuffle] = shuffle_unit_size;
        }

        // auto& MC = proc.MC;
        // MC.init_open(proc.P);
        // for (size_t i = 0; i < tmp.size(); i++)
        //     MC.prepare_open(tmp[i]);
        // MC.exchange(proc.P);
        // for (size_t i = 0; i < tmp.size(); i++)
        //     cout << "Setup tmp[" << i << "]=" << MC.finalize_open() << endl;

        toShuffle.push_back(tmp);
        isExact[currentShuffle] = exact;
    }

    // Apply the shuffles.
    for (size_t pass = 0; pass < n_passes; pass++)
    {
        for (int depth = 0; depth < max_depth; depth++) {
            vector<vector<array<int, 5>>> allIndices;
            proc.protocol.init_mul();

            for (size_t current_shuffle = 0; current_shuffle < n_shuffles; current_shuffle++) {
                int n = toShuffle[current_shuffle].size() / unit_sizes[current_shuffle];
                if (depth >= log2(n)) {
                    allIndices.push_back({});
                    continue;
                }

                const auto isReverse = reverse[current_shuffle];
                size_t configIdx = pass;
                if (isReverse)
                    configIdx = n_passes - pass - 1;

                auto& config = shuffles[current_shuffle][configIdx];

                vector<array<int, 5>> indices = waksman_round_init(
                    toShuffle[current_shuffle],
                    unit_sizes[current_shuffle],
                    depth,
                    config,
                    true,
                    isReverse
                );
                allIndices.push_back(indices);
            }
            proc.protocol.exchange();
            for (size_t current_shuffle = 0; current_shuffle < n_shuffles; current_shuffle++) {
                int n = toShuffle[current_shuffle].size() / unit_sizes[current_shuffle];
                if (depth >= log2(n)) {
                    continue;
                }
                waksman_round_finish(toShuffle[current_shuffle], unit_sizes[current_shuffle], allIndices[current_shuffle]);
            }
        }

        for (int depth = max_depth - 1; depth >= 0; depth--) {
            vector<vector<array<int, 5>>> allIndices;
            proc.protocol.init_mul();

            for (size_t current_shuffle = 0; current_shuffle < n_shuffles; current_shuffle++) {
                int n = toShuffle[current_shuffle].size() / unit_sizes[current_shuffle];
                if (depth > log2(n) - 2) {
                    allIndices.push_back({});
                    continue;
                }

                const auto isReverse = reverse[current_shuffle];
                size_t configIdx = pass;
                if (isReverse)
                    configIdx = n_passes - pass - 1;

                auto& config = shuffles[current_shuffle][configIdx];

                vector<array<int, 5>> indices = waksman_round_init(
                    toShuffle[current_shuffle],
                    unit_sizes[current_shuffle],
                    depth,
                    config,
                    false,
                    isReverse
                );
                allIndices.push_back(indices);
            }
            proc.protocol.exchange();
            for (size_t current_shuffle = 0; current_shuffle < n_shuffles; current_shuffle++) {
                int n = toShuffle[current_shuffle].size() / unit_sizes[current_shuffle];
                if (depth > log2(n) - 2) {
                    continue;
                }
                waksman_round_finish(toShuffle[current_shuffle], unit_sizes[current_shuffle], allIndices[current_shuffle]);
            }
        }
    }

    // Write the shuffled results into memory.
    for (size_t currentShuffle = 0; currentShuffle < n_shuffles; currentShuffle++) {
        const size_t n = sizes[currentShuffle];
        const size_t shuffled_unit_size = unit_sizes[currentShuffle];
        const size_t output_base = destinations[currentShuffle];

        const vector<T>& shuffledData = toShuffle[currentShuffle];

        // auto& MC = proc.MC;
        // MC.init_open(proc.P);
        // for (size_t i = 0; i < shuffledData.size(); i++)
        //     MC.prepare_open(shuffledData[i]);
        // MC.exchange(proc.P);
        // for (size_t i = 0; i < shuffledData.size(); i++)
        //     cout << "Setup shuffledData[" << i << "]=" << MC.finalize_open() << endl;

        if (isExact[currentShuffle])
            for (size_t i = 0; i < n; i++)
                a[output_base + i] = shuffledData[i];
        else
        {
            const size_t original_unit_size = shuffled_unit_size - 1;
            const size_t n_shuffle = n / original_unit_size;
            const size_t n_shuffle_pow2 = shuffledData.size() / shuffled_unit_size;

            // Reveal the "real element" flags.
            auto& MC = proc.MC;
            MC.init_open(proc.P);
            for (size_t i = 0; i < n_shuffle_pow2; i++) {
                MC.prepare_open(shuffledData.at((i + 1) * shuffled_unit_size - 1));
            }
            MC.exchange(proc.P);

            // Filter out the real elements.
            size_t i_shuffle = 0;
            for (size_t i = 0; i < n_shuffle_pow2; i++)
            {
                auto bit = MC.finalize_open();
                if (bit == 1)
                {
                    // only output real elements
                    for (size_t j = 0; j < original_unit_size; j++)
                        a.at(output_base + i_shuffle * original_unit_size + j) =
                                shuffledData.at(i * shuffled_unit_size + j);
                    i_shuffle++;
                }
            }
            if (i_shuffle != n_shuffle)
                throw runtime_error("incorrect shuffle");
        }
    }
}


template<class T>
void SecureShuffle<T>::inverse_permutation(vector<T> &stack, size_t n, size_t output_base,
                                           size_t input_base) {
    int alice = 0;
    int bob = 1;

    auto &P = proc.P;
    auto &input = proc.input;

    // This method only supports two players
    assert(P.num_players() == 2);
    // The current implementation assumes a semi-honest environment
    assert(!T::malicious);

    // We are dealing directly with permutations, so the unit_size will always be 1.
    this->unit_size = 1;
    // We need to account for sizes which are not a power of 2
    size_t n_pow2 = (1u << int(ceil(log2(n))));

    // Copy over the input registers
    pre(stack, n, input_base);
    // Alice generates stack local permutation and shares the waksman configuration bits secretly to Bob.
    vector<int> perm_alice(n_pow2);
    if (P.my_num() == alice)
        perm_alice = generate_random_permutation(n);
    configure(alice, &perm_alice, n);
    // Apply perm_alice to perm_alice to get perm_bob,
    // stack permutation that we can reveal to Bob without Bob learning anything about perm_alice (since it is masked by perm_a)
    iter_waksman(true);
    // Store perm_bob at stack[output_base]
    post(stack, n, output_base);

    // Reveal permutation perm_bob = perm_a * perm_alice
    // Since this permutation is masked by perm_a, Bob learns nothing about perm
    vector<int> perm_bob(n_pow2);
    typename T::PrivateOutput output(proc);
    for (size_t i = 0; i < n; i++)
        output.prepare_sending(stack[output_base + i], bob);
    output.exchange();
    for (size_t i = 0; i < n; i++) {
        // TODO: Is there a better way to convert a T::clear to int?
        bigint val;
        output.finalize(bob).to(val);
        perm_bob[i] = (int) val.get_si();
    }

    vector<int> perm_bob_inv(n_pow2);
    if (P.my_num() == bob) {
        for (int i = 0; i < (int) n; i++)
            perm_bob_inv[perm_bob[i]] = i;
        // Pad the permutation to n_pow2
        // Required when using waksman networks
        for (int i = (int) n; i < (int) n_pow2; i++)
            perm_bob_inv[i] = i;
    }

    // Alice secret shares perm_a with bob
    // perm_a is stored in the stack at output_base
    input.reset_all(P);
    if (P.my_num() == alice) {
        for (int i = 0; i < (int) n; i++)
            input.add_mine(perm_alice[i]);
    }
    input.exchange();
    for (int i = 0; i < (int) n; i++)
        stack[output_base + i] = input.finalize(alice);

    // The two parties now jointly compute perm_a * perm_bob_inv to obtain perm_inv
    pre(stack, n, output_base);
    configure(bob, &perm_bob_inv, n);
    iter_waksman(true);
    // perm_inv is written back to stack[output_base]
    post(stack, n, output_base);
}

template<class T>
void SecureShuffle<T>::pre(vector<T>& a, size_t n, size_t input_base)
{
    n_shuffle = n / unit_size;
    assert(unit_size * n_shuffle == n);
    size_t n_shuffle_pow2 = (1u << int(ceil(log2(n_shuffle))));
    exact = (n_shuffle_pow2 == n_shuffle) or not T::malicious;
    to_shuffle.clear();

    if (exact)
    {
        to_shuffle.resize(n_shuffle_pow2 * unit_size);
        for (size_t i = 0; i < n; i++)
            to_shuffle[i] = a[input_base + i];
    }
    else
    {
        // sorting power of two elements together with indicator bits
        to_shuffle.resize((unit_size + 1) << int(ceil(log2(n_shuffle))));
        for (size_t i = 0; i < n_shuffle; i++)
        {
            for (int j = 0; j < unit_size; j++)
                to_shuffle[i * (unit_size + 1) + j] = a[input_base
                        + i * unit_size + j];
            to_shuffle[i * (unit_size + 1) + unit_size] = T::constant(1,
                    proc.P.my_num(), proc.MC.get_alphai());
        }
        this->unit_size++;
    }
}

template<class T>
void SecureShuffle<T>::post(vector<T>& a, size_t n, size_t output_base)
{
    if (exact)
        for (size_t i = 0; i < n; i++)
            a[output_base + i] = to_shuffle[i];
    else
    {
        auto& MC = proc.MC;
        MC.init_open(proc.P);
        int shuffle_unit_size = this->unit_size;
        int unit_size = shuffle_unit_size - 1;
        for (size_t i = 0; i < to_shuffle.size() / shuffle_unit_size; i++)
            MC.prepare_open(to_shuffle.at((i + 1) * shuffle_unit_size - 1));
        MC.exchange(proc.P);
        size_t i_shuffle = 0;
        for (size_t i = 0; i < n_shuffle; i++)
        {
            auto bit = MC.finalize_open();
            if (bit == 1)
            {
                // only output real elements
                for (int j = 0; j < unit_size; j++)
                    a.at(output_base + i_shuffle * unit_size + j) =
                            to_shuffle.at(i * shuffle_unit_size + j);
                i_shuffle++;
            }
        }
        if (i_shuffle != n_shuffle)
            throw runtime_error("incorrect shuffle");
    }
}

template<class T>
vector<int> SecureShuffle<T>::generate_random_permutation(int n) {
    vector<int> perm;
    int n_pow2 = 1 << int(ceil(log2(n)));
    int shuffle_size = n;
    for (int j = 0; j < n_pow2; j++)
        perm.push_back(j);
    SeededPRNG G;
    for (int i = 0; i < shuffle_size; i++) {
        int j = G.get_uint(shuffle_size - i);
        swap(perm[i], perm[i + j]);
    }

    return perm;
}

template<class T>
void SecureShuffle<T>::player_round(int config_player) {
    vector<int> random_perm(n_shuffle);
    if (proc.P.my_num() == config_player)
        random_perm = generate_random_permutation(n_shuffle);
    configure(config_player, &random_perm, n_shuffle);
    iter_waksman();
}

template<class T>
int SecureShuffle<T>::generate(int n_shuffle, store_type& store)
{
    int res = store.add();
    auto& shuffle = store.get(res);

    for (auto i: proc.protocol.get_relevant_players()) {
        vector<int> perm;
        if (proc.P.my_num() == i)
            perm = generate_random_permutation(n_shuffle);
        configure(i, &perm, n_shuffle);

        shuffle.push_back(config);
    }

    return res;
}

template<class T>
void SecureShuffle<T>::configure(int config_player, vector<int> *perm, int n) {
    auto &P = proc.P;
    auto &input = proc.input;
    input.reset_all(P);
    int n_pow2 = 1 << int(ceil(log2(n)));
    Waksman waksman(n_pow2);

    // The player specified by config_player configures the shared waksman network
    // using its personal permutation
    if (P.my_num() == config_player) {
        auto config_bits = waksman.configure(*perm);
        for (size_t i = 0; i < config_bits.size(); i++) {
            auto &x = config_bits[i];
            for (size_t j = 0; j < x.size(); j++)
                if (waksman.matters(i, j) and not waksman.is_double(i, j))
                    input.add_mine(int(x[j]));
                else if (waksman.is_double(i, j))
                    assert(x[j] == x[j - 1]);
                else
                    assert(x[j] == 0);
        }
        // The other player waits for its share of the configured waksman network
    } else
        for (size_t i = 0; i < waksman.n_bits(); i++)
            input.add_other(config_player);

    input.exchange();
    config.clear();
    typename T::Protocol checker(P);
    checker.init(proc.DataF, proc.MC);
    checker.init_dotprod();
    auto one = T::constant(1, P.my_num(), proc.MC.get_alphai());
    for (size_t i = 0; i < waksman.n_rounds(); i++)
    {
        config.push_back({});
        for (int j = 0; j < n_pow2; j++)
        {
            if (waksman.matters(i, j) and not waksman.is_double(i, j))
            {
                config.back().push_back(input.finalize(config_player));
                if (T::malicious)
                    checker.prepare_dotprod(config.back().back(),
                            one - config.back().back());
            }
            else if (waksman.is_double(i, j))
                config.back().push_back(config.back().back());
            else
                config.back().push_back({});
        }
    }

    if (T::malicious)
    {
        checker.next_dotprod();
        checker.exchange();
        assert(
                typename T::clear(
                        proc.MC.open(checker.finalize_dotprod(waksman.n_bits()),
                                P)) == 0);
        checker.check();
    }
}

template<class T>
void SecureShuffle<T>::waksman(vector<T>& a, int depth, int start)
{
    int n = a.size();

    if (n == 2)
    {
        cond_swap(a[0], a[1], config.at(depth).at(start));
        return;
    }

    vector<T> a0(n / 2), a1(n / 2);
    for (int i = 0; i < n / 2; i++)
    {
        a0.at(i) = a.at(2 * i);
        a1.at(i) = a.at(2 * i + 1);

        cond_swap(a0[i], a1[i], config.at(depth).at(i + start + n / 2));
    }

    waksman(a0, depth + 1, start);
    waksman(a1, depth + 1, start + n / 2);

    for (int i = 0; i < n / 2; i++)
    {
        a.at(2 * i) = a0.at(i);
        a.at(2 * i + 1) = a1.at(i);
        cond_swap(a[2 * i], a[2 * i + 1], config.at(depth).at(i + start));
    }
}

template<class T>
void SecureShuffle<T>::cond_swap(T& x, T& y, const T& b)
{
    auto diff = proc.protocol.mul(x - y, b);
    x -= diff;
    y += diff;
}

template<class T>
void SecureShuffle<T>::iter_waksman(bool reverse)
{
    int n = to_shuffle.size() / unit_size;

    for (int depth = 0; depth < log2(n); depth++)
        waksman_round(depth, true, reverse);

    for (int depth = log2(n) - 2; depth >= 0; depth--)
        waksman_round(depth, false, reverse);
}

template<class T>
void SecureShuffle<T>::waksman_round(int depth, bool inwards, bool reverse)
{
    int n = to_shuffle.size() / unit_size;
    assert((int) config.at(depth).size() == n);
    int nblocks = 1 << depth;
    int size = n / (2 * nblocks);
    bool outwards = !inwards;
    proc.protocol.init_mul();
    vector<array<int, 5>> indices;
    indices.reserve(n / 2);
    Waksman waksman(n);
    for (int k = 0; k < n / 2; k++)
    {
        int j = k % size;
        int i = k / size;
        int base = 2 * i * size;
        int in1 = base + j + j * inwards;
        int in2 = in1 + inwards + size * outwards;
        int out1 = base + j + j * outwards;
        int out2 = out1 + outwards + size * inwards;
        int i_bit = base + j + size * (outwards ^ reverse);
        bool run = waksman.matters(depth, i_bit);
        if (run)
        {
            for (int l = 0; l < unit_size; l++)
                proc.protocol.prepare_mul(config.at(depth).at(i_bit),
                        to_shuffle.at(in1 * unit_size + l)
                                - to_shuffle.at(in2 * unit_size + l));
        }
        indices.push_back({{in1, in2, out1, out2, run}});
    }
    proc.protocol.exchange();
    tmp.resize(to_shuffle.size());
    for (int k = 0; k < n / 2; k++)
    {
        auto idx = indices.at(k);
        for (int l = 0; l < unit_size; l++)
        {
            T diff;
            if (idx[4])
                diff = proc.protocol.finalize_mul();
            tmp.at(idx[2] * unit_size + l) = to_shuffle.at(
                    idx[0] * unit_size + l) - diff;
            tmp.at(idx[3] * unit_size + l) = to_shuffle.at(
                    idx[1] * unit_size + l) + diff;
        }
    }
    swap(tmp, to_shuffle);
}

template<class T>
vector<array<int, 5>> SecureShuffle<T>::waksman_round_init(vector<T> &toShuffle, size_t shuffle_unit_size, int depth, vector<vector<T>> &iter_waksman_config, bool inwards, bool reverse) {
    int n = toShuffle.size() / shuffle_unit_size;
    assert((int) iter_waksman_config.at(depth).size() == n);
    int n_blocks = 1 << depth;
    int size = n / (2 * n_blocks);
    bool outwards = !inwards;
    vector<array<int, 5>> indices;
    indices.reserve(n / 2);
    Waksman waksman(n);
    for (int k = 0; k < n / 2; k++)
    {
        int j = k % size;
        int i = k / size;
        int base = 2 * i * size;
        int in1 = base + j + j * inwards;
        int in2 = in1 + inwards + size * outwards;
        int out1 = base + j + j * outwards;
        int out2 = out1 + outwards + size * inwards;
        int i_bit = base + j + size * (outwards ^ reverse);
        bool run = waksman.matters(depth, i_bit);
        if (run)
        {
            for (size_t l = 0; l < shuffle_unit_size; l++)
                proc.protocol.prepare_mul(iter_waksman_config.at(depth).at(i_bit),
                        toShuffle.at(in1 * shuffle_unit_size + l) - toShuffle.at(in2 * shuffle_unit_size + l));
        }
        indices.push_back({in1, in2, out1, out2, run});
    }
    return indices;
}

template<class T>
void SecureShuffle<T>::waksman_round_finish(vector<T> &toShuffle, size_t unit_size, vector<array<int, 5>> indices) {
    int n = toShuffle.size() / unit_size;

    vector<T> tmp(toShuffle.size());
    for (int k = 0; k < n / 2; k++)
    {
        auto idx = indices.at(k);
        for (size_t l = 0; l < unit_size; l++)
        {
            T diff;
            if (idx[4])
                diff = proc.protocol.finalize_mul();
            tmp.at(idx[2] * unit_size + l) = toShuffle.at(
                    idx[0] * unit_size + l) - diff;
            tmp.at(idx[3] * unit_size + l) = toShuffle.at(
                    idx[1] * unit_size + l) + diff;
        }
    }

    // for (size_t i = 0; i < toShuffle.size(); i++)
        // toShuffle[i] = tmp.at(i);
    swap(tmp, toShuffle);
}


#endif /* PROTOCOLS_SECURESHUFFLE_HPP_ */
