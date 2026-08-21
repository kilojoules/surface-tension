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
    # We want distance % M == 0.
    
    # Calculate prefix sums modulo M. 
    # P[i] = sum(A[0...i-1]) % M. P[0] = 0.
    # We use accumulate to avoid explicit loops.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of A modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Let x = P[s-1] and y = P[t-1].
    # If s < t: (y - x) % M == 0  =>  y % M == x % M
    # If s > t: (total_sum_mod - x + y) % M == 0 => (x - y) % M == total_sum_mod % M
    
    # Count occurrences of each prefix sum modulo M for indices 0 to N-1
    # Note: P has N+1 elements, we only need the first N (P[0] to P[N-1])
    counts = Counter(P[:N])
    
    # For a fixed x = P[s-1], we need y = P[t-1] such that:
    # 1. s < t and y == x (mod M)
    # 2. s > t and y == (x - total_sum_mod) (mod M)
    
    # Total pairs (s, t) where s < t and P[s-1] == P[t-1] (mod M):
    # For each value v, if it appears C times, there are C*(C-1)/2 pairs.
    # However, the problem asks for pairs (s, t). 
    # Let's evaluate the condition for each s:
    # For a fixed s, we need t such that:
    # (t > s and P[t-1] == P[s-1]) OR (t < s and P[t-1] == (P[s-1] - total_sum_mod) % M)
    
    # Let target_y(x) = (x - total_sum_mod) % M
    # Total pairs = Sum_{x in P[0...N-1]} [ (count(x) - 1) if we only consider t > s ]
    # This is tricky because the s < t and s > t conditions are asymmetric.
    
    # Let's use the property:
    # Total = Sum_{x} (count(x) * (count(x) - 1) / 2)  <-- this is for s < t and P[s-1] == P[t-1]
    # PLUS Sum_{x} (count(x) * count(target_y(x))) <-- this is for s > t
    # BUT we must subtract cases where s > t and P[t-1] == (P[s-1] - total_sum_mod) % M 
    # AND P[t-1] == P[s-1] (which happens if total_sum_mod == 0).
    
    # Correct logic:
    # For each s in {1...N}, we seek t in {1...N}, t != s such that:
    # If t > s: P[t-1] ≡ P[s-1] (mod M)
    # If t < s: P[t-1] ≡ P[s-1] - total_sum_mod (mod M)
    
    # Let C(v) be the number of i in {0...N-1} such that P[i] == v.
    # Total pairs = Sum_{v=0 to M-1} [ C(v) * (C(v) - 1) / 2 ]  <-- pairs (s, t) with s < t
    #              + Sum_{v=0 to M-1} [ C(v) * C((v - total_sum_mod) % M) ] <-- pairs (s, t) with s > t
    # Wait, the second term includes cases where t < s. 
    # For a fixed s, the number of t < s such that P[t-1] == (P[s-1] - total_sum_mod) % M
    # is the number of indices i < s-1 such that P[i] == (P[s-1] - total_sum_mod) % M.
    
    # Let's reconsider:
    # We want pairs (s, t) with s != t such that dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) % total_lake_length if t > s
    # dist(s, t) = (total_lake_length - P[s-1] + P[t-1]) % total_lake_length if t < s
    
    # Let L = total_sum_mod.
    # Condition: 
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] ≡ P[s-1] (mod M)
    # If s > t: (L - P[s-1] + P[t-1]) % M == 0 => P[t-1] ≡ (P[s-1] - L) (mod M)
    
    # Let's use the counts:
    # For each value v, there are C(v) indices.
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == v: C(v) * (C(v) - 1) / 2
    # Pairs (s, t) with s > t and P[t-1] == (P[s-1] - L) % M:
    # This is Sum_{s=1 to N} (count of i < s-1 where P[i] == (P[s-1] - L) % M)
    
    # Let's use a different approach for the s > t part:
    # Sum_{s > t} [P[t-1] == (P[s-1] - L) % M]
    # = Sum_{v} [ C(v) * C((v - L) % M) ] - Sum_{s=t} [P[s-1] == (P[s-1] - L) % M]
    # This is not quite right because the Sum_{v} includes t > s.
    
    # Let's use the symmetry:
    # Total = Sum_{s < t} [P[t-1] == P[s-1]] + Sum_{s > t} [P[t-1] == (P[s-1] - L) % M]
    # Let f(v) = C(v).
    # Part 1: Sum_{v} f(v)(f(v)-1)/2
    # Part 2: Sum_{s=1 to N} (count of i < s-1 such that P[i] == (P[s-1] - L) % M)
    
    # To calculate Part 2 without loops, we can use the fact that:
    # Sum_{s > t} [P[t-1] == (P[s-1] - L) % M] 
    # = Sum_{v} (count of i such that P[i] == v) * (count of j such that P[j] == (v + L) % M and j < i)
    
    # If L == 0:
    # Part 1: Sum f(v)(f(v)-1)/2
    # Part 2: Sum f(v)(f(v)-1)/2
    # Total: Sum f(v)(f(v)-1)
    
    # If L != 0:
    # Part 1: Sum f(v)(f(v)-1)/2
    # Part 2: Sum_{s > t} [P[t-1] == (P[s-1] - L) % M]
    # Notice that for L != 0, the condition P[t-1] == (P[s-1] - L) % M 
    # implies P[t-1] != P[s-1].
    # Thus, the sets of pairs {(s, t) | s < t, P[s-1] == P[t-1]} and 
    # {(s, t) | s > t, P[t-1] == (P[s-1] - L) % M} are disjoint.
    
    # For Part 2, we can use the property:
    # Sum_{s > t} [P[t-1] == (P[s-1] - L) % M] + Sum_{s < t} [P[t-1] == (P[s-1] - L) % M]
    # = Sum_{v} C(v) * C((v - L) % M)
    # And Sum_{s < t} [P[t-1] == (P[s-1] - L) % M] is the same as 
    # Sum_{t > s} [P[t-1] == (P[s-1] - L) % M].
    
    # This is getting complex. Let's use a simpler observation:
    # We want pairs (s, t) such that (P[t-1] - P[s-1]) % M == 0 (for s < t)
    # or (Total - P[s-1] + P[t-1]) % M == 0 (for s > t).
    # This is equivalent to:
    # s < t: P[t-1] ≡ P[s-1] (mod M)
    # s > t: P[t-1] ≡ P[s-1] - Total (mod M)
    
    # Let's use the property that for any two indices i, j (i < j):
    # Pair (s=i+1, t=j+1) is valid if P[j] ≡ P[i] (mod M)
    # Pair (s=j+1, t=i+1) is valid if P[i] ≡ P[j] - Total (mod M)
    
    # Total = Sum_{i < j} [P[j] == P[i]] + Sum_{i < j} [P[i] == (P[j] - Total) % M]
    # Total = Sum_{v} C(v)(C(v)-1)/2 + Sum_{i < j} [P[i] == (P[j] - Total) % M]
    
    # For the second term:
    # If Total == 0: Sum_{i < j} [P[i] == P[j]] = Sum C(v)(C(v)-1)/2
    # If Total != 0: 
    # Sum_{i < j} [P[i] == (P[j] - Total) % M] 
    # We can't easily get this from C(v) without the order.
    # Wait, the problem says N is 2e5 and M is 1e6. A loop is forbidden, but 
    # we can use a generator expression inside sum().
    
    # Let's use the fact that we can iterate through the array once.
    # We need Sum_{j=1 to N-1} (count of i < j such that P[i] == (P[j] - Total) % M)
    # We can't use a loop to update the counter. 
    # But we can use a different trick.
    # The total number of pairs (i, j) with i < j such that P[i] == (P[j] - Total) % M
    # is (Sum_{v} C(v) * C((v + Total) % M) - Sum_{i} [P[i] == (P[i] - Total) % M]) / 2
    # ONLY IF the condition is symmetric. It's not.
    
    # Actually, the most direct way to calculate Sum_{i <