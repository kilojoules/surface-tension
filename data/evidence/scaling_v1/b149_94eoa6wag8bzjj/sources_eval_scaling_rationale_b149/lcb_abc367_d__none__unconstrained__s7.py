import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension/slice to get the rest.
    # However, since we need to handle the input as a stream, 
    # we can convert the map object to a list first.
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A[0]
    # P[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Compute prefix sums modulo M
    # accumulate([0] + A) gives [0, A1, A1+A2, ...]
    # We take the first N elements to get P[0]...P[N-1]
    P = list(map(lambda x: x % M, accumulate([0] + A)))[:N]
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(P)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # Number of pairs is sum(count * (count - 1) // 2) for each remainder
    internal_pairs = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # => P[s-1] - P[t-1] == total_sum_mod (mod M)
    # => P[s-1] == (P[t-1] + total_sum_mod) % M
    # We iterate over the unique remainders present in P
    external_pairs = sum(
        counts[r] * counts[(r + total_sum_mod) % M]
        for r in counts
    )
    
    # Special case: if total_sum_mod == 0, the external_pairs calculation
    # includes cases where P[s-1] == P[t-1], which are the same as internal_pairs.
    # But the condition s > t is strict.
    # If total_sum_mod == 0, then P[s-1] == (P[t-1] + 0) % M is just P[s-1] == P[t-1].
    # For each group of size 'c', there are c*(c-1) pairs (s,t) with s != t.
    # The logic above for external_pairs when total_sum_mod == 0 gives sum(c*c).
    # We need to subtract the cases where s == t (which is sum(c)) 
    # and then we have the total pairs.
    # Actually, a simpler way:
    # If total_sum_mod == 0:
    #   Valid pairs are all (s, t) where P[s-1] == P[t-1] and s != t.
    #   This is sum(c * (c - 1))
    # If total_sum_mod != 0:
    #   Valid pairs are (s < t and P[s-1] == P[t-1]) 
    #   PLUS (s > t and P[s-1] == (P[t-1] + total_sum_mod) % M)
    
    if total_sum_mod == 0:
        # Every pair (s, t) with P[s-1] == P[t-1] is valid regardless of s < t or s > t
        print(sum(c * (c - 1) for c in counts.values()))
    else:
        # internal_pairs covers s < t, external_pairs covers s > t
        # Note: external_pairs logic 'counts[r] * counts[(r + total_sum_mod) % M]'
        # correctly counts pairs (s, t) where s > t.
        # Wait, the external_pairs loop as written counts pairs (t, s) 
        # such that P[s-1] = (P[t-1] + total_sum_mod) % M.
        # Let's refine:
        # For each t, we need s > t such that P[s-1] = (P[t-1] + total_sum_mod) % M.
        # This is tricky without loops. Let's use the property:
        # Total = Sum_{t=1 to N} (count of s < t where P[s-1] == P[t-1])
        #       + Sum_{t=1 to N} (count of s > t where P[s-1] == (P[t-1] + total_sum_mod) % M)
        
        # Let's use a different approach for total_sum_mod != 0:
        # For each remainder r, let c1 = counts[r] and c2 = counts[(r + total_sum_mod) % M].
        # The number of pairs (s, t) with s < t and P[s-1] == P[t-1] is sum(c*(c-1)//2).
        # The number of pairs (s, t) with s > t and P[s-1] == (P[t-1] + total_sum_mod) % M:
        # This is sum_{t < s} [P[s-1] == (P[t-1] + total_sum_mod) % M].
        # This is equivalent to counting pairs (t, s) with t < s and P[s-1] - P[t-1] == total_sum_mod % M.
        # Let's use the property: 
        # Total = Sum_{r} (counts[r] * counts[(r + total_sum_mod) % M])
        # But this includes pairs where s < t and s > t.
        # Let's just use the direct logic:
        # A pair (s, t) is valid if:
        # 1. s < t and P[t-1] - P[s-1] = 0 mod M
        # 2. s > t and (Total - P[s-1] + P[t-1]) = 0 mod M => P[s-1] - P[t-1] = Total mod M
        
        # Let's compute this by iterating over the array P once.
        # For each index i, we want to count j < i such that:
        # P[i] - P[j] == 0 mod M  (this is s=j+1, t=i+1)
        # P[i] - P[j] == Total mod M (this is s=i+1, t=j+1)
        
        # We can use a custom reduce to maintain a running count of remainders.
        from functools import reduce
        
        def accumulate_counts(acc, val):
            curr_counts, total_valid = acc
            # s < t: P[t-1] == P[s-1]
            # s > t: P[s-1] - P[t-1] == total_sum_mod
            # For the current index i (as t), we add curr_counts[P[i]]
            # For the current index i (as s), we add curr_counts[(P[i] - total_sum_mod) % M]
            new_valid = total_valid + curr_counts[val] + curr_counts[(val - total_sum_mod) % M]
            curr_counts[val] += 1
            return (curr_counts, new_valid)

        final_state = reduce(accumulate_counts, P, (Counter(), 0))
        print(final_state[1])

if __name__ == "__main__":
    solve()