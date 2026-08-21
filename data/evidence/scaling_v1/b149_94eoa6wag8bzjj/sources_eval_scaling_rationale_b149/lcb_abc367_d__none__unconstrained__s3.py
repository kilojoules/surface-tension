The constraint to avoid explicit loops and recursion forces a shift from imperative state management to a functional paradigm, emphasizing data transformations over step-by-step instructions. To solve this problem, I will use `itertools.accumulate` to compute prefix sums of the distances, list comprehensions for data processing, and `collections.Counter` to efficiently count occurrences of remainders modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Compute prefix sums modulo M
    # P will have N elements: P[0] is distance from 1 to 1 (0), P[N-1] is distance from 1 to N.
    P = list(accumulate(A[:-1], lambda x, y: (x + y) % M, initial=0))
    total_sum = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(P)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # Number of pairs is sum(count * (count - 1) // 2) for each remainder.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (total_sum - P[s-1] + P[t-1]) % M == 0
    # => P[t-1] == (P[s-1] - total_sum) % M
    # We iterate over all possible remainders r = P[s-1]
    # The required P[t-1] is (r - total_sum) % M.
    # Note: s > t means we are looking for pairs of indices (i, j) where i > j.
    # This is equivalent to summing counts[r] * counts[(r - total_sum) % M]
    # But we must exclude cases where s=t (though the problem says s != t).
    # Since we need s > t, we can't simply use the combinations formula.
    # Instead, we realize that for every s, we need t < s such that P[t-1] == (P[s-1] - total_sum) % M.
    # However, a simpler way:
    # Total pairs (s, t) is the sum over all r of:
    # count(r) * count((r - total_sum) % M)
    # Then we subtract the cases where s = t (which happens if total_sum % M == 0).
    
    # Let's refine:
    # We want pairs (s, t) such that dist(s, t) % M == 0.
    # If s < t: P[t-1] - P[s-1] \equiv 0 mod M  => P[t-1] \equiv P[s-1] mod M.
    # If s > t: total_sum - P[s-1] + P[t-1] \equiv 0 mod M => P[t-1] \equiv (P[s-1] - total_sum) mod M.
    
    # Let C[r] be the number of i in {0, ..., N-1} such that P[i] == r.
    # Pairs (s, t) with s < t: sum_{r} C[r]*(C[r]-1)//2
    # Pairs (s, t) with s > t: 
    # For each s in {1, ..., N}, we need t < s such that P[t-1] == (P[s-1] - total_sum) % M.
    # This is harder to do without a loop if we don't know the distribution.
    # Actually, we can use the property:
    # Total pairs = sum_{r=0 to M-1} (C[r] * C[(r - total_sum) % M])
    # This sum counts all pairs (s, t) such that dist(s, t) % M == 0, 
    # INCLUDING cases where s=t (if total_sum % M == 0) AND 
    # it treats the "s < t" and "s > t" logic differently.
    
    # Let's use the direct logic:
    # 1. s < t: P[t-1] == P[s-1]. Number of pairs: sum(C[r]*(C[r]-1)//2)
    # 2. s > t: P[t-1] == (P[s-1] - total_sum) % M.
    #    This is sum_{r} (C[r] * C[(r - total_sum) % M]) 
    #    BUT this includes cases where the index of P[t-1] is >= index of P[s-1].
    #    That's not quite right.
    
    # Correct approach:
    # The condition is: (P[t-1] - P[s-1]) % M == 0 if s < t
    # (total_sum - P[s-1] + P[t-1]) % M == 0 if s > t.
    # Let's just use the property that for any s, t:
    # dist(s, t) = (P[t-1] - P[s-1]) % total_sum_of_A
    # We want (P[t-1] - P[s-1]) % M == 0 if s < t
    # and (total_sum + P[t-1] - P[s-1]) % M == 0 if s > t.
    
    # Let's use a different approach:
    # For each r in 0...M-1, let C[r] be the count of P[i] == r.
    # For a fixed r, there are C[r] indices.
    # Any two indices i < j with P[i] == P[j] == r give a pair (s, t) = (i+1, j+1).
    # Any two indices i > j with P[i] == r and P[j] == (r - total_sum) % M give a pair (s, t) = (i+1, j+1).
    
    # Let's calculate:
    # Part 1: s < t => P[s-1] == P[t-1]. 
    # Count: sum(C[r] * (C[r] - 1) // 2 for r in range(M))
    
    # Part 2: s > t => P[t-1] == (P[s-1] - total_sum) % M.
    # This is sum_{i=0 to N-1} (count of j < i such that P[j] == (P[i] - total_sum) % M).
    # To do this without loops, we can use a trick with the total counts.
    # Total pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    # sum_{r=0 to M-1} (C[r] * C[(r - total_sum) % M])
    # Wait, if total_sum % M == 0, then dist(s, t) % M == 0 iff P[s-1] == P[t-1].
    # In that case, there are N * (N-1) pairs if all P are same, or sum(C[r]*(C[r]-1)).
    # If total_sum % M != 0, then for any s, there is exactly one remainder r' such that
    # if P[t-1] == r', then dist(s, t) % M == 0.
    # Specifically, if s < t, we need P[t-1] == P[s-1].
    # If s > t, we need P[t-1] == (P[s-1] - total_sum) % M.
    
    # Let's use the most robust method:
    # For each r, we have C[r] occurrences.
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == r: C[r]*(C[r]-1)//2
    # Pairs (s, t) with s > t and P[s-1] == r and P[t-1] == (r - total_sum) % M:
    # This is tricky because of the s > t condition.
    # Let's use the fact:
    # Total pairs (s, t) with s != t is:
    # sum_{r=0 to M-1} [ C[r] * C[r] - (C[r] if total_sum % M == 0 else 0) ]
    # No, that's not right.
    
    # Let's go back:
    # For each i in 0...N-1:
    #   We want j > i such that P[j] == P[i]  (s=i+1, t=j+1)
    #   We want j < i such that P[j] == (P[i] - total_sum) % M (s=i+1, t=j+1)
    
    # Let's use the property:
    # Total = sum_{r=0 to M-1} (C[r] * C[(r - total_sum) % M])
    # This sum counts pairs (i, j) such that P[j] == (P[i] - total_sum) % M.
    # If total_sum % M == 0, this is sum(C[r]^2). These are pairs where P[i] == P[j].
    #   - If i < j, dist is P[j]-P[i] == 0 mod M. (Correct)
    #   - If i > j, dist is total_sum + P[j]-P[i] == 0 mod M. (Correct)
    #   - If i == j, dist is 0, but s != t. (Incorrect, must subtract)
    #   - So if total_sum % M == 0, ans = sum(C[r]^2) - N.
    # If total_sum % M != 0, then r != (r - total_sum) % M.
    #   - For any pair (i, j), it's impossible for both (i < j and P[i]==P[j]) 
    #     AND (i > j and P[j] == (P[i] - total_sum) % M) to be true for the same (i, j).
    #   - Actually, for any pair {i, j} with i < j:
    #     Either dist(i+1, j+1) % M == 0 (if P[i] == P[j])
    #     Or dist(j+1, i+1) % M == 0 (if P[i] == (P[j] - total_sum) % M)
    #     Or neither.
    #   - The sum sum_{r=0 to M-1} (C[r] * C[(r - total_sum) % M]) 
    #     counts pairs (i, j) such that P[j] == (P[i] - total_sum) % M.
    #     If i < j, then dist(i+1, j+1) = P[j] - P[i] = -total_sum % M.
    #     If i > j, then dist(i+1, j+1) = total_sum + P[j] - P[i] = 0 % M.
    #     Wait, this is simpler:
    #     The pair (s, t) with s > t is valid iff P[t-1] == (P[s-1] - total_sum) % M.
    #     The pair (s, t) with s < t is valid iff P[t-1] == P[s-1].
    
    # Final Logic:
    # Ans