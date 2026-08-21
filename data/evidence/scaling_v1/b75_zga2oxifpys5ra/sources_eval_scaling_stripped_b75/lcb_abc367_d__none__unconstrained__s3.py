import sys
from functools import reduce

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A1 % M, P[2] = (A1 + A2) % M, ...
    # We use reduce to avoid loops and maintain a functional style
    # state: (current_prefix_sum, list_of_prefix_sums)
    def accumulate_mod(state, val):
        curr, acc = state
        next_val = (curr + val) % M
        return (next_val, acc + [next_val])

    _, prefixes = reduce(accumulate_mod, A, (0, [0]))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1]) % M
    # We want (Total_Sum - P[s-1] + P[t-1]) % M == 0
    
    # Let L = Total_Sum % M. 
    # For s < t: P[t-1] == P[s-1]
    # For s > t: P[t-1] == (P[s-1] - L) % M
    
    # Note: prefixes list has N+1 elements. The last one is Total_Sum % M.
    # We only care about P[0]...P[N-1] because s and t are in 1...N.
    P = prefixes[:-1]
    L = prefixes[-1]
    
    # Count occurrences of each remainder
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    for x in P:
        counts[x] += 1
        
    # For a fixed s, we need t such that:
    # 1. t > s and P[t-1] == P[s-1]
    # 2. t < s and P[t-1] == (P[s-1] - L) % M
    
    # Total pairs = Sum_{v=0 to M-1} (count[v] * (count[v] - 1) // 2) 
    # This covers s < t where P[s-1] == P[t-1].
    # However, the "s > t" case depends on L.
    
    # Let's use a different approach:
    # For every s, we seek t != s such that dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) % M if t > s
    # dist(s, t) = (L - P[s-1] + P[t-1]) % M if t < s
    
    # If L == 0:
    # Both conditions become P[t-1] == P[s-1].
    # Total = Sum(count[v] * (count[v] - 1))
    
    # If L != 0:
    # For a fixed s, we need t > s with P[t-1] == P[s-1]
    # OR t < s with P[t-1] == (P[s-1] - L) % M.
    # This is harder to sum globally. Let's use the property:
    # Total = Sum_{s=1 to N} [ (count of t > s where P[t-1]==P[s-1]) 
    #                        + (count of t < s where P[t-1]==(P[s-1]-L)%M) ]
    
    # Let's track the running count of each remainder as we iterate s from 1 to N.
    # left_counts[v]: number of t < s such that P[t-1] == v
    # right_counts[v]: number of t > s such that P[t-1] == v
    
    # Initial right_counts is the total counts
    # We can't use loops, so we use a map/reduce or a generator with a mutable state.
    # Since we must avoid loops, we can use a generator to process the counts.
    
    # To avoid loops and maintain state, we can use a helper function with a list for counts.
    def calculate_pairs(p_vals, total_counts, mod, length):
        # We need to track how many of each remainder we've seen so far
        # state: (current_left_counts, total_pairs)
        # But we can't use a loop to update current_left_counts.
        # Actually, we can use a list and mutate it inside a function called by map/reduce.
        
        left_counts = [0] * mod
        
        def process(p):
            # For current s, P[s-1] = p
            # t > s: P[t-1] == p. Count is (total_counts[p] - left_counts[p] - 1)
            # t < s: P[t-1] == (p - L) % M. Count is left_counts[(p - L) % M]
            
            # We use a list to bypass the closure restriction on scalars
            res = (total_counts[p] - left_counts[p] - 1) + left_counts[(p - L) % mod]
            left_counts[p] += 1
            return res

        return sum(map(process, p_vals))

    print(calculate_pairs(P, counts, M, N))

if __name__ == "__main__":
    solve()