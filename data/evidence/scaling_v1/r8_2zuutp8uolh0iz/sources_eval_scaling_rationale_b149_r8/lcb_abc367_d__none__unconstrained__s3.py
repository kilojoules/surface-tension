The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without `for` or `while` loops, I will use `itertools.accumulate` to compute prefix sums of the distances, and `collections.Counter` combined with list comprehensions to count the occurrences of each remainder modulo $M$.

The core logic is:
1. Calculate the prefix sum $P_i$ of the distances $A$.
2. The distance from $s$ to $t$ (where $s < t$) is $(P_{t-1} - P_{s-1}) \pmod M$. This is $0$ if $P_{t-1} \equiv P_{s-1} \pmod M$.
3. The distance from $s$ to $t$ (where $s > t$) is the total sum minus the distance from $t$ to $s$.
4. By using the prefix sums modulo $M$, we can count pairs using the frequency of each remainder.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN-1]
    # We use accumulate to get prefix sums and map them to modulo M
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (TotalSum - (P[s-1] - P[t-1])) % M
    
    # Prefix sums modulo M
    # P_vals will have N elements: distance from area 1 to area 1, 2, ..., N
    P_vals = list(map(lambda x: x % M, accumulate(A, initial=0)[:-1]))
    total_sum_mod = sum(A) % M
    
    # Count frequencies of each remainder
    counts = Counter(P_vals)
    
    # For a fixed s and t (s < t), distance is 0 mod M if P[t-1] == P[s-1]
    # Number of pairs (s, t) with s < t is sum(count * (count - 1) // 2)
    ans_s_less_t = sum([count * (count - 1) // 2 for count in counts.values()])
    
    # For s > t, distance is (total_sum_mod + P[t-1] - P[s-1]) % M
    # We need (P[s-1] - P[t-1]) % M == total_sum_mod
    # Let P[s-1] = x and P[t-1] = y. We need (x - y) % M == total_sum_mod
    # Which means y % M == (x - total_sum_mod) % M
    
    # We iterate over the unique remainders present in P_vals
    # For each x, the number of y's is counts[x] * counts[(x - total_sum_mod) % M]
    # However, we must exclude cases where s = t (though the problem says s != t)
    # and we must handle the case where total_sum_mod == 0 carefully.
    
    # To avoid loops, we use a list comprehension over the keys of the counter.
    # For s > t, we are looking for pairs (s, t) such that 
    # dist(s, t) = (Total - dist(t, s)) % M = 0
    # Total % M = dist(t, s) % M
    # total_sum_mod = (P[s-1] - P[t-1]) % M
    
    # Let's calculate the number of pairs (s, t) with s > t such that 
    # P[s-1] - P[t-1] \equiv total_sum_mod (mod M)
    # This is equivalent to P[t-1] \equiv (P[s-1] - total_sum_mod) (mod M)
    
    # We can sum counts[x] * counts[(x - total_sum_mod) % M] for all x in counts
    # But this counts pairs (s, t) where s and t can be anything.
    # We specifically need s > t.
    # Actually, a simpler way:
    # For every pair (s, t) with s < t, we checked if dist(s, t) % M == 0.
    # Now we check if dist(t, s) % M == 0, which is (Total - dist(s, t)) % M == 0.
    # This is equivalent to dist(s, t) % M == Total % M.
    
    # Let x = P[t-1] and y = P[s-1] with s < t.
    # dist(s, t) % M = (x - y) % M.
    # We want (x - y) % M == total_sum_mod.
    
    # If total_sum_mod == 0:
    # Then dist(s, t) % M == 0 is the same as dist(t, s) % M == 0.
    # The number of pairs is simply 2 * ans_s_less_t.
    # Wait, if total_sum_mod == 0, then (x-y)%M == 0 implies x == y.
    # The number of pairs (s, t) with s != t is N * (N-1) if all P are same and M=1.
    # Let's use the property:
    # For each pair {s, t} with s < t:
    # Pair (s, t) is valid if (P[t-1] - P[s-1]) % M == 0
    # Pair (t, s) is valid if (total_sum_mod - (P[t-1] - P[s-1])) % M == 0
    
    # Let diff = (P[t-1] - P[s-1]) % M.
    # We want to count pairs (s, t) with s < t where diff == 0
    # PLUS pairs (s, t) with s < t where diff == total_sum_mod.
    
    # If total_sum_mod == 0, these two conditions are the same.
    # But the problem asks for pairs (s, t), and (s, t) is different from (t, s).
    # If total_sum_mod == 0, then dist(s, t) % M == 0 <=> dist(t, s) % M == 0.
    # So we just multiply ans_s_less_t by 2.
    
    # If total_sum_mod != 0, the two conditions are mutually exclusive.
    # We need to count pairs (s, t) with s < t such that (P[t-1] - P[s-1]) % M == total_sum_mod.
    # This is sum(counts[x] * counts[(x - total_sum_mod) % M]) 
    # But we need to be careful about the s < t constraint.
    
    # Let's redefine:
    # We want to count pairs (s, t) with 1 <= s, t <= N and s != t such that
    # clockwise distance from s to t is 0 mod M.
    # Let P[i] be the distance from 1 to i+1.
    # For s < t: dist = P[t-1] - P[s-1]
    # For s > t: dist = (P[N-1] + A[N-1]) - (P[s-1] - P[t-1])
    
    # Let's use the property:
    # Total pairs = sum_{x, y} count(x)*count(y) where (x - y) % M == 0 (for s < t)
    #              + sum_{x, y} count(x)*count(y) where (total_sum_mod - (x - y)) % M == 0 (for s > t)
    # This is not quite right because the s < t constraint is handled by the combination.
    
    # Correct logic:
    # For every pair of indices i, j from {0, ..., N-1} with i < j:
    # Pair (i+1, j+1) is valid if (P[j] - P[i]) % M == 0
    # Pair (j+1, i+1) is valid if (total_sum_mod - (P[j] - P[i])) % M == 0
    
    # Let's count how many pairs (i, j) with i < j have (P[j] - P[i]) % M == v
    # for v = 0 and v = total_sum_mod.
    
    # To count pairs (i, j) with i < j such that P[j] - P[i] \equiv v (mod M):
    # This is equivalent to P[j] \equiv P[i] + v (mod M).
    # We can't use a loop, so we use the fact that:
    # Total pairs (i, j) with i != j such that P[j] - P[i] \equiv v (mod M)
    # is sum(counts[x] * counts[(x - v) % M])
    # If v == 0, we subtract the cases where i == j: sum(counts[x])
    # Then we divide by 2? No, because the order (i, j) matters.
    
    # Let's use the property:
    # The number of pairs (s, t) with s < t such that dist(s, t) % M == 0 
    # is the number of pairs (i, j) with i < j such that P[j] == P[i].
    # This is sum(c * (c-1) // 2).
    
    # The number of pairs (s, t) with s > t such that dist(s, t) % M == 0
    # is the number of pairs (i, j) with i < j such that (total_sum_mod - (P[j] - P[i])) % M == 0
    # which is (P[j] - P[i]) % M == total_sum_mod.
    
    # How to count pairs i < j such that P[j] - P[i] \equiv v (mod M) without loops?
    # Let's use the fact that:
    # sum_{i < j} [P[j] - P[i] \equiv v] + sum_{i > j} [P[j] - P[i] \equiv v] 
    # = sum_{i != j} [P[j] - P[i] \equiv v]
    # And sum_{i > j} [P[j] - P[i] \equiv v] = sum_{i < j} [P[i] - P[j] \equiv v]
    # = sum_{i < j} [P[j] - P[i] \equiv -v]
    
    # So, let f(v) be the number of pairs i < j such that (P[j] - P[i]) % M == v.
    # sum_{x} counts[x] * counts[(x - v) % M] = f(v) + f(-v) if v != 0
    # and if v == 0, it is 2 * f(0) + N.
    
    # This is getting complex. Let's use a simpler observation:
    # We want to count pairs (s, t) such that:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (total_sum_mod - (P[s-1] - P[t-1])) % M == 0
    
    # Condition 1: P[t-1] \equiv P[s-1] (mod M)
    # Condition 2: P[s-1] - P[t-1] \equiv total_sum_mod (mod M)
    
    # Let's just use the counts:
    # For Condition 1: sum(c * (c-1) // 2 for c in counts.values())
    # For Condition 2: 
    # We need to count pairs (i, j) with i < j such that P[i] - P[j] \equiv total_sum_mod (mod M)
