import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # Total number of good sequences S is (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    # Since we cannot use loops, we use reduce to build the sequence.
    
    # Precompute total S
    # S = (N*K)! / (K!**N)
    # However, we only need to compare the target index against the number of 
    # sequences starting with a certain digit.
    
    # The number of ways to arrange the remaining items is:
    # (sum(counts))! / product(counts[i]!)
    # We can compute this using a helper function.
    def count_permutations(counts):
        total = sum(counts)
        # Using a property: ways = comb(total, counts[0]) * comb(total-counts[0], counts[1])...
        # But since we need to avoid loops, we use reduce.
        return reduce(lambda acc, c: acc * comb(total - (sum(counts[:counts.index(c)]) if counts.index(c) > 0 else 0), c), 
                      range(len(counts)), 1)
    
    # The above count_permutations is slightly wrong because of the index() logic and range().
    # Let's redefine it using a more robust reduce.
    def get_total_ways(counts):
        # ways = (sum counts)! / product(c!)
        # We use the multiplicative formula for multinomial coefficients.
        return reduce(lambda a, b: a * comb(b[0], b[1]), 
                      [(sum(counts), c) for c in counts], 1)
    
    # Wait, the above logic is still slightly flawed. Let's use the standard multinomial:
    # Ways = comb(n, k1) * comb(n-k1, k2) * ...
    def get_multinomial(counts):
        return reduce(lambda acc, x: (acc[0] * comb(x[0], x[1]), x[0] - x[1]), 
                      [(sum(counts), c) for c in counts], (1, sum(counts)))[0]

    # Calculate S
    S = get_multinomial([K] * N)
    target = (S + 1) // 2

    # We need to find the target-th sequence.
    # State: (current_counts, current_target, result_sequence)
    # We use reduce to simulate the process of picking the digit for each position.
    
    def pick_digit(state, _):
        counts, target_idx, seq = state
        
        # We need to find the smallest digit d such that 
        # sum_{i=1}^{d-1} ways(counts with digit i decremented) < target_idx
        # and sum_{i=1}^{d} ways(counts with digit i decremented) >= target_idx
        
        # To avoid loops, we use a list comprehension to calculate ways for each possible digit
        # only for digits that still have counts > 0.
        
        # ways_for_digit[d] = get_multinomial(counts updated by picking d)
        # We only care about d in 1...N
        
        def find_digit(t_idx):
            # Calculate ways for each digit 1 to N
            # If counts[d-1] == 0, ways = 0
            ways = [get_multinomial(
                [counts[i] - (1 if i == d-1 else 0) for i in range(N)]
            ) if counts[d-1] > 0 else 0 for d in range(1, N + 1)]
            
            # Find the digit d where the cumulative sum reaches target_idx
            # We use a helper to find the index
            def search(cum_sum, d):
                if d > N: return N, t_idx
                if cum_sum + ways[d-1] >= t_idx:
                    return d, t_idx - cum_sum
                return search(cum_sum + ways[d-1], d + 1)
            
            # Since we can't use recursion/loops, we use a trick with a list 
            # to find the first index where prefix sum >= target
            prefix_sums = list(reduce(lambda acc, x: acc + [acc[-1] + x], ways, [0]))
            # prefix_sums[d] is the sum of ways for digits 1 to d.
            # We want the smallest d such that prefix_sums[d] >= t_idx.
            # We can use a list comprehension to find all d that satisfy this, then take the min.
            d = min([d for d in range(1, N + 1) if prefix_sums[d] >= t_idx])
            return d, t_idx - prefix_sums[d-1]

        digit, new_target = find_digit(target_idx)
        
        # Update counts
        new_counts = [counts[i] - (1 if i == digit - 1 else 0) for i in range(N)]
        return (new_counts, new_target, seq + [digit])

    # Initial state: (counts, target, sequence)
    initial_state = ([K] * N, target, [])
    final_state = reduce(pick_digit, range(N * K), initial_state)
    
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()