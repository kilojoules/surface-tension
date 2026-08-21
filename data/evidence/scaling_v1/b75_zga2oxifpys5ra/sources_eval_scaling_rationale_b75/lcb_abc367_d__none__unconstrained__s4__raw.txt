The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional style. While idiomatic Python often uses `for` loops for readability, utilizing high-order functions and comprehensions can lead to more concise, declarative code that describes *what* to calculate rather than *how* to step through it. For this problem, I will use `itertools.accumulate` to calculate prefix sums of distances, list comprehensions for transformations, and `collections.Counter` to efficiently count occurrences of remainders, allowing the final answer to be computed via a `sum()` over the frequency distribution.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator/iterator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances: P_i is distance from rest area 1 to i+1
    # P_0 = 0, P_1 = A_1, P_2 = A_1 + A_2, ..., P_N = sum(A_i)
    # We only need these values modulo M
    p = [x % m for x in accumulate(a, initial=0)]
    
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) % M
    # We want (P_{t-1} - P_{s-1}) % M == 0, which means P_{t-1} % M == P_{s-1} % M
    # The distance from s to t (s > t) is (Total_Sum - P_{s-1} + P_{t-1}) % M
    # We want (Total_Sum - P_{s-1} + P_{t-1}) % M == 0
    
    # Let S = Total_Sum % M = P[N]
    s_total = p[n]
    
    # Count frequencies of each remainder in P_0, ..., P_{N-1}
    # P[N] is the total sum, we only consider indices 0 to N-1 for s and t
    counts = Counter(p[:n])
    
    # For a fixed remainder r, let count(r) be the number of i in {0, ..., N-1} 
    # such that P_i % M == r.
    # 1. For s < t: we need P_{t-1} == P_{s-1} (mod M).
    #    Number of pairs is sum(count(r) * (count(r) - 1) // 2)
    # 2. For s > t: we need P_{t-1} == (P_{s-1} - S) (mod M).
    #    Number of pairs is sum(count(r) * count((r - s_total) % m))
    #    However, we must exclude cases where s = t (though the problem says s != t).
    #    Wait, the logic for s > t is: 
    #    Dist(s, t) = (P_{N} - P_{s-1}) + P_{t-1}
    #    We want (S - P_{s-1} + P_{t-1}) % M == 0  => P_{t-1} % M == (P_{s-1} - S) % M
    
    # To avoid loops, we use generator expressions inside sum()
    # Part 1: s < t
    ans_lt = sum(v * (v - 1) // 2 for v in counts.values())
    
    # Part 2: s > t
    # For each r, we look for the count of remainder (r - s_total) % m
    # We must ensure we don't count the case where s=t, but s > t already handles that.
    # Note: if s_total % m == 0, then (r - s_total) % m == r, and we are counting 
    # pairs (s, t) where P_{s-1} == P_{t-1}. Since s > t, this is distinct from s < t.
    ans_gt = sum(v * counts[(r - s_total) % m] for r, v in counts.items())
    
    # Special case: if s_total % m == 0, the logic above for s > t counts 
    # pairs where P_{s-1} == P_{t-1}. If s_total % m == 0, then 
    # Dist(s, t) = (0 - P_{s-1} + P_{t-1}) % M. 
    # For this to be 0, P_{s-1} must equal P_{t-1}.
    # The sum(v * counts[(r - s_total) % m]) correctly counts all pairs (s, t) 
    # such that s > t and the condition holds.
    # However, if s_total % m == 0, then (r - s_total) % m == r.
    # The term v * counts[r] includes the case where s-1 and t-1 are the same index.
    # But the constraint is s != t. Since we are iterating over the distribution,
    # we must subtract the cases where the index of s is the same as t.
    # Actually, the logic `sum(v * counts[(r - s_total) % m])` counts pairs of indices.
    # If s_total % m == 0, it counts pairs (i, j) where P_i == P_j.
    # This includes i == j. Since we need s > t, we only care about i != j.
    # If s_total % m == 0, the number of pairs (i, j) with i > j and P_i == P_j 
    # is exactly the same as the number of pairs with i < j and P_i == P_j.
    
    # Correct logic for s > t:
    # For each i in {0, ..., N-1}, we need j in {0, ..., N-1} such that 
    # j < i and P_j % M == (P_i - S) % M.
    # This is harder without a loop. Let's use the property:
    # Total pairs (s, t) with s != t is:
    # For each s, we need t != s such that Dist(s, t) % M == 0.
    # Dist(s, t) = (P_{t-1} - P_{s-1}) % M if s < t
    # Dist(s, t) = (P_N - P_{s-1} + P_{t-1}) % M if s > t
    
    # Let's redefine:
    # We want (P_{t-1} - P_{s-1}) % M == 0 for s < t
    # We want (P_{t-1} - (P_{s-1} - S)) % M == 0 for s > t
    
    # Let's use the property that we can just iterate over all remainders r in 0..M-1
    # For a fixed r, let c[r] be the count of i in {0..N-1} such that P_i % M == r.
    # Pairs (s, t) with s < t: sum(c[r] * (c[r]-1) // 2)
    # Pairs (s, t) with s > t: 
    # We need P_{t-1} % M == (P_{s-1} - S) % M.
    # For a fixed s, the number of t < s is the number of j < s-1 such that P_j == (P_{s-1} - S) % M.
    # This requires knowing the distribution of P values *before* index s.
    # That suggests a loop. But wait!
    # The total number of pairs (s, t) with s != t such that Dist(s, t) % M == 0 is:
    # For each s, we need t such that:
    # 1. t > s and P_{t-1} % M == P_{s-1} % M
    # 2. t < s and P_{t-1} % M == (P_{s-1} - S) % M
    
    # Let's use the fact that we can't use loops, but we can use map/sum/comprehensions.
    # To handle the s > t case without a loop, we can't easily track "running" counts.
    # However, we can use the total counts and subtract the "s < t" cases from the "s > t" logic?
    # No, that's not right.
    
    # Let's reconsider:
    # Total = Sum_{s=1 to N} [ Count(t > s : P_{t-1} == P_{s-1}) + Count(t < s : P_{t-1} == P_{s-1} - S) ]
    # Total = Sum_{r=0 to M-1} [ c[r]*(c[r]-1)//2 + Sum_{s: P_{s-1}=r} Count(t < s : P_{t-1} == r - S) ]
    # The second term is Sum_{s=1 to N} Count(t < s : P_{t-1} == P_{s-1} - S).
    # This is exactly the number of pairs (t, s) with t < s such that P_{t-1} == (P_{s-1} - S) % M.
    # This can be calculated by iterating over all r:
    # For a fixed r, we have c[r] indices where P_i == r, and c[(r-S)%M] indices where P_j == (r-S)%M.
    # We want to count pairs (j, i) such that j < i and P_j == (r-S)%M and P_i == r.
    
    # If S % M != 0, then r != (r-S)%M. The condition j < i doesn't matter 
    # if we just want the total count across all s, t? 
    # No, the condition is specifically s > t.
    # Actually, if S % M != 0, then for any pair (j, i) with P_j == (r-S)%M and P_i == r,
    # either j < i or j > i.
    # This is still tricky. Let's use the property:
    # Total = Sum_{i < j} [I(P_i == P_j) + I(P_i == (P_j - S)%M)]
    # = Sum_{r} [c[r]*(c[r]-1)//2] + Sum_{i < j} [I(P_i == (P_j - S)%M)]
    
    # To calculate Sum_{i < j} [I(P_i == (P_j - S)%M)] without a loop:
    # We can use the fact that:
    # Sum_{i < j} I(P_i == r1 and P_j == r2) + Sum_{i > j} I(P_i == r1 and P_j == r2) 
    # = c[r1] * c[r2] (if r1 != r2)
    # And if r1 == r2, the sum is c[r1] * (c[r1] - 1).
    
    # This doesn't help because we don't know the relative order.
    # Wait, the problem can be solved by doubling the array A.
    # The distance from s to t is simply the sum of A_i from s to t-1.
    # If we have the prefix sums P_0, ..., P_{2N-1}, then 
    # Dist(s, t) = P_{t-1} - P_{s-1} for s < t <= N.
    # For s > t, the distance is P_{t+N-1} - P_{s-1}.
    # In both cases, we want P_{end} - P_{start} \equiv 0 (mod M) where end - start < N.
    
    # Let's use the property:
    # Total = Sum_{i=0 to N-1} Sum_{j=i+1 to i+N-1} [ (P_j - P_i) % M == 0 ]
    # where P is the prefix sum array of A extended to 2N.
    # (P_j - P_i) % M == 0  => P_j % M == P_i % M.
    # So for each i, we need the number of j in {i+1, ..., i+N-1} such