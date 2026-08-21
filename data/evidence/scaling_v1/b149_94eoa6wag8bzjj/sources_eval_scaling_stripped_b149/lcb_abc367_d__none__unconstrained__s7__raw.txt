```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A.
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M.
    # The distance from s to t (s > t) is (TotalSum - P_{s-1} + P_{t-1}) mod M.
    
    # Calculate prefix sums modulo M
    # P[i] = (A_1 + ... + A_i) % M
    # We use accumulate to get [P_1, P_2, ..., P_N]
    # We prepend 0 to represent P_0
    P = [0] + list(accumulate(A, lambda x, y: (x + y) % M))
    
    # Total sum of all A_i modulo M
    total_sum = P[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # For s > t: (total_sum - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_sum (mod M)
    
    # Let's count occurrences of each remainder in P[0...N-1]
    # Note: P[N] is the total sum, but the distance from s to t involves indices 0 to N-1.
    # The rest areas are 1...N. The distance from s to t is:
    # If s < t: sum(A[s-1]...A[t-2]) = P[t-1] - P[s-1]
    # If s > t: sum(A[s-1]...A[N-1]) + sum(A[0]...A[t-2]) = (P[N] - P[s-1]) + P[t-1]
    
    # We only care about P[0] to P[N-1]
    counts = Counter(P[:N])
    
    # For s < t, we need P[t-1] == P[s-1]. 
    # For each remainder r, if it appears C times, there are C*(C-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If P[t-1] == P[s-1], then distance(s, t) is a multiple of M.
    # This covers all s < t.
    
    # For s > t, we need P[s-1] - P[t-1] == total_sum (mod M).
    # This is equivalent to P[s-1] == (P[t-1] + total_sum) (mod M).
    
    # Let's calculate the total pairs:
    # 1. Pairs (s, t) with s < t and P[s-1] == P[t-1]
    # 2. Pairs (s, t) with s > t and P[s-1] == (P[t-1] + total_sum) % M
    
    # Part 1: s < t
    # For each remainder r, there are counts[r] indices.
    # The number of pairs (s, t) with s < t is combinations(counts[r], 2).
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t
    # We need P[s-1] == (P[t-1] + total_sum) % M.
    # Let r1 = P[t-1] and r2 = P[s-1]. We need r2 == (r1 + total_sum) % M.
    # Since s > t, for every pair of indices (i, j) with i < j, 
    # we check if P[j] == (P[i] + total_sum) % M.
    # This is slightly different from Part 1 because the condition depends on total_sum.
    
    # If total_sum % M == 0, then the condition is P[s-1] == P[t-1].
    # This is symmetric to Part 1.
    # If total_sum % M != 0, then P[s-1] and P[t-1] must be different.
    
    if total_sum % M == 0:
        # s > t and P[s-1] == P[t-1]
        # This is the same as s < t and P[s-1] == P[t-1]
        ans_s_gt_t = ans_s_lt_t
    else:
        # s > t and P[s-1] == (P[t-1] + total_sum) % M
        # We need to count pairs (i, j) such that i < j and P[j] == (P[i] + total_sum) % M.
        # This can be done by iterating through the prefix sums and keeping track of counts.
        # But we can't use loops. We can use a trick with Counter.
        # The total number of pairs (i, j) such that P[j] == (P[i] + total_sum) % M
        # is sum(counts[r] * counts[(r + total_sum) % M] for r in counts).
        # However, this includes both i < j and i > j.
        # Let's use the property that for any two distinct indices i, j:
        # Either (P[j] - P[i]) % M == 0 or (P[j] - P[i]) % M == total_sum % M
        # is NOT necessarily true.
        
        # Correct approach for s > t:
        # We want pairs (i, j) such that 0 <= i < j < N and P[j] == (P[i] + total_sum) % M.
        # We can use a generator expression with a helper function or 
        # use the fact that we can compute this using a custom reduce or 
        # by processing the list. Since we can't use loops, we use a 
        # list comprehension that builds a running count.
        # But wait, we can't use loops to build the running count.
        # Let's use the mathematical property:
        # Total pairs (i, j) with i != j such that P[j] - P[i] == total_sum (mod M)
        # is sum(counts[r] * counts[(r + total_sum) % M] for r in counts).
        # For any pair {i, j} with i < j:
        # Distance(i+1, j+1) is (P[j] - P[i]) % M
        # Distance(j+1, i+1) is (total_sum - (P[j] - P[i])) % M
        # We want Distance == 0 (mod M).
        # Case 1: (P[j] - P[i]) % M == 0  => P[j] == P[i] (mod M)
        # Case 2: (total_sum - (P[j] - P[i])) % M == 0 => P[j] - P[i] == total_sum (mod M)
        
        # So the answer is:
        # sum(c*(c-1)//2 for c in counts.values()) + 
        # sum(counts[r] * counts[(r + total_sum) % M] for r in counts if r != (r + total_sum) % M)
        # Wait, the second term is for s > t. 
        # If P[j] - P[i] == total_sum (mod M) for i < j, then distance(j+1, i+1) == 0 (mod M).
        # This is exactly what we need.
        # But the condition P[j] - P[i] == total_sum (mod M) doesn't imply i < j.
        # Actually, for any two indices i, j, if P[j] - P[i] == total_sum (mod M),
        # then distance(j+1, i+1) == 0 (mod M) if i < j.
        # And if i > j, then distance(i+1, j+1) == 0 (mod M) if P[j] - P[i] == total_sum (mod M).
        # This is confusing. Let's simplify:
        # A pair (s, t) is valid if:
        # 1. s < t and P[t-1] - P[s-1] \equiv 0 (mod M)
        # 2. s > t and P[N] - P[s-1] + P[t-1] \equiv 0 (mod M)
        
        # Condition 1: P[t-1] \equiv P[s-1] (mod M)
        # Condition 2: P[s-1] - P[t-1] \equiv P[N] (mod M)
        
        # Let C(r) be the count of i \in {0, ..., N-1} such that P[i] == r (mod M).
        # Pairs for Cond 1: \sum_{r} C(r)(C(r)-1)/2
        # Pairs for Cond 2: \sum_{r} C(r) * C((r - P[N]) % M)
        # BUT, we must ensure s > t. 
        # For a fixed pair of indices {i, j} with i < j:
        # It satisfies Cond 1 if P[j] == P[i].
        # It satisfies Cond 2 if P[j] - P[i] == -P[N] == P[N] (mod M) is FALSE.
        # Cond 2 is P[s-1] - P[t-1] == P[N] (mod M) with s > t.
        # Let i = t-1 and j = s-1. Then i < j and P[j] - P[i] == P[N] (mod M).
        
        # So the total count is:
        # \sum_{r} [C(r)(C(r)-1)/2] + \sum_{i < j} [P[j] - P[i] \equiv P[N] (mod M)]
        
        # How to calculate \sum_{i < j} [P[j] - P[i] \equiv P[N] (mod M)] without loops?
        # We can use the fact that:
        # \sum_{i < j} [P[j] - P[i] \equiv K] + \sum_{i > j} [P[j] - P[i] \equiv K] + \sum_{i=j} [P[j] - P[i] \equiv K]
        # = \sum_{i, j} [P[j] - P[i] \equiv K]
        # The last term is N if K == 0, else 0.
        # The first two terms are symmetric if we replace K with -K.
        # \sum_{i < j} [P[j] - P[i] \equiv K] = \sum_{i > j} [P[i] - P[j] \equiv K]
        # = \sum_{i > j} [P[i] - P[j] \equiv K]
        
        # Let's use the property:
        # Total pairs (i, j) with i != j such that P[j] - P[i] \equiv K (mod M)
        # is \sum_{r} C(r) * C((r - K) % M), subtracting cases where i=j (which happens if K=0).
        # This sum equals: \sum_{i < j} [P[j] - P[i] \equiv K] + \sum_{i > j} [P[j] - P[i] \equiv K]
        # \sum_{i > j} [P[j] - P[i] \equiv K] = \sum_{i < j} [P[i] - P[j] \equiv K] = \sum_{i < j} [P[j] - P[i] \equiv -K]
        
        # So, \sum_{i < j} [P[j] - P[i] \