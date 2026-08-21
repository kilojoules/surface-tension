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
    
    # Let P_i be the prefix sum of A.
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M.
    # The distance from s to t (s > t) is (TotalSum - P_{s-1} + P_{t-1}) mod M.
    
    # Calculate prefix sums modulo M. 
    # P[i] = sum(A[0...i-1]) % M. P[0] = 0.
    # We use accumulate to avoid explicit loops.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of A modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # For s > t: (total_sum_mod - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_sum_mod (mod M)
    
    # Let x = P[s-1] and y = P[t-1].
    # We need to count pairs (s, t) with s != t and 1 <= s, t <= N.
    # This is equivalent to counting pairs (i, j) with 0 <= i, j < N and i != j such that:
    # If i < j: y - x = 0 (mod M)
    # If i > j: total_sum_mod + y - x = 0 (mod M)
    
    # Let's use a frequency map of P[0...N-1]
    counts = Counter(P[:N])
    
    # For a fixed value v, there are counts[v] indices i where P[i] = v.
    # Pairs (i, j) with i < j and P[i] == P[j] contribute:
    # sum(c * (c - 1) // 2 for c in counts.values())
    
    # For pairs (i, j) with i > j and P[i] - P[j] == total_sum_mod (mod M):
    # This is sum(counts[v] * counts[(v - total_sum_mod) % M]) 
    # But we must subtract cases where i == j (which happens if total_sum_mod == 0).
    # Also, the condition i > j is tricky with just counts.
    
    # Alternative approach:
    # Total pairs (s, t) is the sum over all i, j in {0...N-1}, i != j:
    # [i < j and P[j] == P[i]] + [i > j and P[i] - P[j] == total_sum_mod]
    
    # Let S = total_sum_mod.
    # If S == 0:
    # The condition becomes [i < j and P[j] == P[i]] + [i > j and P[i] == P[j]]
    # Which is simply all pairs (i, j) where P[i] == P[j] and i != j.
    # Result: sum(c * (c - 1) for c in counts.values())
    
    # If S != 0:
    # We need i < j and P[j] == P[i]  OR  i > j and P[i] - P[j] == S.
    # Let's evaluate sum_{i < j} [P[j] == P[i]] + sum_{i > j} [P[i] - P[j] == S].
    # The first term is sum(c * (c - 1) // 2 for c in counts.values()).
    # The second term: for each i, we need j < i such that P[j] == (P[i] - S) % M.
    # This can be solved by iterating through P and keeping track of counts of seen values.
    
    # To avoid loops, we can use a mathematical trick for the second term:
    # sum_{i > j} [P[i] - P[j] == S] = sum_{v} (count(v) * count((v - S) % M))
    # MINUS the cases where the index of v is actually less than the index of (v-S).
    # Actually, the simplest way to think about it:
    # For every pair of values (v, (v-S)%M), one is the "start" and one is the "end".
    # In a circle, for any two distinct indices i, j, exactly one of (i,j) or (j,i) 
    # is a "clockwise" path.
    # The distance from s to t is (P[t-1] - P[s-1]) mod M if s < t
    # and (Total - P[s-1] + P[t-1]) mod M if s > t.
    
    # Let x = P[s-1] and y = P[t-1].
    # Condition: 
    # If s < t: y - x = 0 mod M
    # If s > t: Total + y - x = 0 mod M => x - y = Total mod M
    
    # Let's use the property:
    # Total pairs = sum_{i < j} [P[j] == P[i]] + sum_{i > j} [P[i] - P[j] == Total]
    
    # Let's compute the second term using a list comprehension and sum():
    # We need sum_{i=1 to N-1} (count of P[j] == (P[i] - Total) % M for j < i)
    # This is still a loop. Let's use the frequency map.
    # For any two distinct indices i, j:
    # If P[i] == P[j], then one of (i,j) or (j,i) satisfies the condition IF Total == 0.
    # If Total == 0, both (i,j) and (j,i) satisfy it.
    # If Total != 0, and P[i] == P[j], then only the pair (s,t) with s < t satisfies it.
    # If P[i] - P[j] == Total (mod M), then only the pair (s,t) with s > t satisfies it.
    
    # Correct Logic:
    # Pair (s, t) is valid if:
    # 1. s < t and P[t-1] ≡ P[s-1] (mod M)
    # 2. s > t and P[t-1] ≡ P[s-1] - Total (mod M)
    
    # Let C(v) be the number of times v appears in P[0...N-1].
    # Term 1: sum_{v} C(v) * (C(v) - 1) / 2
    # Term 2: sum_{i > j} [P[j] ≡ P[i] - Total (mod M)]
    
    # To calculate Term 2 without a loop:
    # It is sum_{v} (C(v) * C((v - Total) % M)) 
    # MINUS sum_{i < j} [P[j] ≡ P[i] - Total (mod M)]
    
    # This is getting complex. Let's use the most direct observation:
    # We want pairs (i, j) with 0 <= i, j < N, i != j such that:
    # (P[j] - P[i]) % M == 0 if i < j
    # (Total + P[j] - P[i]) % M == 0 if i > j
    
    # This is equivalent to:
    # Count pairs (i, j) with i < j and P[i] == P[j]
    # + Count pairs (i, j) with i > j and P[i] - P[j] == Total (mod M)
    
    # Let's use the fact that for any i != j, 
    # if P[i] == P[j], then (i, j) is valid if i < j, and (j, i) is valid if Total == 0.
    # if P[i] - P[j] == Total (mod M), then (j, i) is valid if i > j.
    
    # Let's simplify:
    # The answer is sum_{v=0 to M-1} (C(v) * C((v - Total) % M))
    # BUT we must exclude cases where i == j.
    # If i == j, then (P[i] - P[i]) % M = 0.
    # This is only a "valid" distance if Total == 0.
    # However, the problem says s != t.
    # If Total == 0, then P[i] == P[j] implies both (i, j) and (j, i) are valid.
    # If Total != 0, then P[i] == P[j] implies only (i, j) with i < j is valid.
    # And P[i] - P[j] == Total implies only (j, i) with i > j is valid.
    
    # Wait, if Total != 0:
    # For any pair {i, j} with i < j:
    # - (i, j) is valid if P[j] - P[i] == 0 mod M
    # - (j, i) is valid if Total + P[i] - P[j] == 0 mod M => P[j] - P[i] == Total mod M
    # These two conditions are mutually exclusive since Total != 0.
    
    # So if Total != 0:
    # Ans = sum_{v} (C(v) * C(v) - C(v)) / 2  <-- for P[j] == P[i]
    #     + sum_{v} (C(v) * C((v - Total) % M)) <-- for P[i] - P[j] == Total, but this includes i=j?
    # No, if Total != 0, then v != (v - Total) % M, so i cannot be j.
    # But the second term sum_{v} C(v) * C((v - Total) % M) counts all pairs (i, j) 
    # such that P[i] - P[j] == Total. 
    # For each such pair, either i > j or i < j.
    # If i > j, it's a valid pair (s, t) = (i+1, j+1).
    # If i < j, it's NOT a valid pair.
    
    # Let's use the property:
    # Total valid = sum_{i < j} [P[i] == P[j]] + sum_{i > j} [P[i] - P[j] == Total]
    # Let's use the symmetry:
    # sum_{i > j} [P[i] - P[j] == Total] = sum_{i < j} [P[j] - P[i] == -Total]
    # = sum_{i < j} [P[j] - P[i] == (M - Total) % M]
    
    # So Ans = sum_{i < j} ([P[j] - P[i] == 0] + [P[j] - P[i] == (M - Total) % M])
    # If (M - Total) % M == 0 (i.e., Total == 0):
    # Ans = sum_{i < j} (2 * [P[j] == P[i]]) = sum(C(v) * (C(v) - 1) for v in counts)
    # If (M - Total) % M != 0:
    # Ans = sum_{i < j} [P[j] == P[i]] + sum_{i < j} [P[j] - P[i] == (M - Total) % M]
    # The first term is sum(C(v) * (C(v) - 1) // 2 for v in counts)
    # The second term is sum_{j} (C(P[j]) * C((P