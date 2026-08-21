The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without `for` or `while` loops, I will use `itertools.accumulate` to calculate prefix sums of the distances, and `collections.Counter` combined with generator expressions to count the occurrences of each remainder modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances: P[i] is distance from rest area 1 to i+1
    # P[0] = 0, P[1] = A_1, P[2] = A_1 + A_2, ...
    # We use accumulate to avoid loops.
    p = list(accumulate(a, lambda x, y: x + y, initial=0))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] % M == P[s-1] % M
    # For s > t, the distance is (Total_Sum - P[s-1]) + P[t-1]
    # This is (P[N] - P[s-1] + P[t-1]) % M == 0
    
    # Let R[i] = P[i] % M
    r = [val % m for val in p]
    
    # Count occurrences of each remainder
    counts = Counter(r)
    
    # For a fixed remainder v, if there are k indices i such that R[i] == v:
    # 1. Pairs (s, t) with s < t: we choose 2 indices from k. 
    #    Number of pairs is k * (k - 1) / 2.
    # 2. Pairs (s, t) with s > t: 
    #    The condition is (R[N] - R[s-1] + R[t-1]) % M == 0
    #    Which is R[t-1] % M == (R[s-1] - R[N]) % M
    
    # Let's refine:
    # For every pair (s, t) with s != t:
    # If s < t: distance is P[t-1] - P[s-1]
    # If s > t: distance is P[N] - P[s-1] + P[t-1]
    
    # Let X = P[N] % M.
    # If s < t, we need R[t-1] == R[s-1]
    # If s > t, we need R[t-1] == (R[s-1] - X) % M
    
    x = r[n]
    
    # We can calculate the total pairs using a generator expression and sum()
    # For each remainder v that appears k times:
    # The number of pairs (s, t) with s < t and R[s-1] == R[t-1] == v is k*(k-1)//2
    # However, we must exclude the index N (the total sum) from the s < t logic 
    # because the problem says s, t are in {1, ..., N}.
    # The indices for R are 0 to N. R[0] is area 1, R[N-1] is area N.
    # So we only care about R[0]...R[N-1].
    
    r_limited = r[:n]
    counts_lim = Counter(r_limited)
    
    # Pairs (s, t) with s < t:
    # For each v, k = counts_lim[v], pairs = k*(k-1)//2
    ans_st = sum(k * (k - 1) // 2 for k in counts_lim.values())
    
    # Pairs (s, t) with s > t:
    # We need R[t-1] == (R[s-1] - X) % M
    # For each s-1 in 0...N-1, we need to count t-1 in 0...s-2 such that R[t-1] == (R[s-1] - X) % M
    # This is harder without a loop. Let's use the property:
    # Total pairs (s, t) is the sum over all v:
    # count(R[i] == v) * count(R[j] == (v - X) % M)
    # But we must subtract the cases where the condition s > t is not met.
    # Actually, the simplest way:
    # For every pair (s, t) with s != t:
    # If s < t, dist = P[t-1] - P[s-1]
    # If s > t, dist = P[N] - (P[s-1] - P[t-1])
    # Notice that (P[t-1] - P[s-1]) % M == 0  <=>  (P[N] - (P[s-1] - P[t-1])) % M == X
    
    # Let's use the property:
    # For a fixed pair {s, t} with s < t:
    # Clockwise s to t is (P[t-1] - P[s-1])
    # Clockwise t to s is (P[N] - (P[t-1] - P[s-1]))
    # We want to know if (P[t-1] - P[s-1]) % M == 0 OR (P[N] - (P[t-1] - P[s-1])) % M == 0
    
    # Let d = (P[t-1] - P[s-1]) % M.
    # We want d == 0 (for s -> t) or (X - d) % M == 0 (for t -> s).
    # Note: if X == 0, then d == 0 implies both are multiples of M.
    # If X != 0, then d == 0 and d == X are mutually exclusive.
    
    # Total pairs = sum_{v} (counts_lim[v] * counts_lim[(v - X) % M])
    # But this counts pairs (s, t) where R[t-1] - R[s-1] == X.
    # Let's be precise:
    # For each pair i, j in {0, ..., N-1} with i < j:
    # s=i+1, t=j+1: distance is P[j] - P[i]. Multiple of M if R[j] == R[i].
    # s=j+1, t=i+1: distance is P[N] - (P[j] - P[i]). Multiple of M if R[j] - R[i] == X % M.
    
    # Total = sum_{v} (counts_lim[v] * counts_lim[(v + X) % M])
    # Wait, if X == 0, the condition R[j] == R[i] and R[j] - R[i] == X are the same.
    # If X == 0, for each v, we have k*(k-1) pairs.
    # If X != 0, for each v, we have k_v * k_{v+X} pairs.
    
    # Correct logic:
    # For each pair i < j:
    # Pair (i+1, j+1) is valid if R[j] - R[i] \equiv 0 (mod M)
    # Pair (j+1, i+1) is valid if R[j] - R[i] \equiv X (mod M)
    
    # If X == 0:
    # Both are valid if R[j] == R[i].
    # For each v, k=counts_lim[v], we get k*(k-1) pairs.
    # If X != 0:
    # (i+1, j+1) is valid if R[j] == R[i]
    # (j+1, i+1) is valid if R[j] == (R[i] + X) % M
    # Total = sum_{v} (k_v * (k_v - 1) // 2) + sum_{v} (k_v * k_{(v+X)%M})
    # Note: the second sum is over all v, but we need i < j.
    # Actually, the second sum is simpler:
    # For any two distinct indices i, j, the distance from s to t is a multiple of M if:
    # (s < t and R[t-1] == R[s-1]) OR (s > t and R[t-1] == (R[s-1] - X) % M)
    
    # Let's use the most robust method:
    # For every pair i, j in {0, ..., N-1} with i != j:
    # If i < j, we check R[j] == R[i]
    # If i > j, we check R[j] == (R[i] - X) % M
    
    # This is equivalent to:
    # Sum_{i < j} [R[j] == R[i]] + Sum_{i > j} [R[j] == (R[i] - X) % M]
    # = Sum_{v} (k_v * (k_v - 1) // 2) + Sum_{i, j: i > j} [R[j] == (R[i] - X) % M]
    
    # To calculate the second term without loops:
    # We need to count pairs (i, j) such that i > j and R[j] == (R[i] - X) % M.
    # This is equivalent to counting pairs (i, j) such that i != j and R[j] == (R[i] - X) % M,
    # then handling the i < j cases.
    # But the condition R[j] == (R[i] - X) % M is different from R[j] == R[i] unless X == 0.
    
    # Let's use:
    # Total = Sum_{v} (k_v * (k_v - 1) // 2) + Sum_{i=0 to N-1} (count of j < i such that R[j] == (R[i] - X) % M)
    # The second term can be computed by iterating through the list and keeping track of counts.
    # Since we can't use loops, we can use a trick with a custom function in accumulate or a similar tool.
    # However, the simplest way to count pairs (i, j) with i > j and R[j] == target is:
    # For a fixed v, we want to count pairs (i, j) such that i > j, R[i] == v, and R[j] == (v - X) % M.
    # This is not simply k_v * k_{v-X} because of the i > j constraint.
    # WAIT: The distance from s to t is (P[t-1] - P[s-1]) if s < t.
    # If s > t, the distance is (P[N] - P[s-1]) + P[t-1].
    # Let's use the property:
    # Distance(s, t) = (P[t-1] - P[s-1]) % P[N]
    # We want (P[t-1] - P[s-1]) % M == 0 (if s < t)
    # OR (P[N] - P[s-1] + P[t-1]) % M == 0 (if s > t)
    
    # Let R[i] = P[i] % M.
    # s < t: R[t-1] - R[s-1] \equiv 0 (mod M)  => R[t-1] == R[s-1]
    # s > t: X - R[s-1] + R[t-1] \equiv 0 (mod M) => R[t-1] == (R[s-1] - X) % M
    
    # Let's use a generator to compute the second term:
    # We need Sum_{i=0 to N-1} (count