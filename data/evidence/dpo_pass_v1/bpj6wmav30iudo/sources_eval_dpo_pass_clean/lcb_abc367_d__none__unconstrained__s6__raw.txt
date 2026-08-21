The constraint to avoid explicit `for` and `while` loops forces a shift from imperative programming to a functional style. While explicit loops are often more readable for complex state changes, using comprehensions and built-in higher-order functions can lead to concise, expression-oriented code that leverages Python's internal optimizations. For this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances and `collections.Counter` to efficiently count the occurrences of each remainder modulo $M$.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and map to integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use a trick to unpack the iterator
    # We can't use a for loop, but we can use a list comprehension 
    # or map to consume the input.
    
    # To handle the input without a for loop, we convert the map to a list
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Prefix sums: P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A[0], A[0]+A[1], ...]
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # initial=0 ensures P[0] = 0
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of A modulo M
    total_sum_mod = P[N]
    
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (total_sum_mod - P[s-1] + P[t-1]) % M == 0 for s > t
    
    # Let's analyze the condition:
    # For a fixed s, we need t such that:
    # If t > s: P[t-1] ≡ P[s-1] (mod M)
    # If t < s: P[t-1] ≡ P[s-1] - total_sum_mod (mod M)
    
    # Let counts be the frequency of each remainder in P[0...N-1]
    # P[N] is the total sum, we only consider indices 0 to N-1 for s and t
    counts = Counter(P[:N])
    
    # For each remainder r, there are counts[r] positions.
    # Any pair (s, t) with s < t and P[s-1] == P[t-1] works.
    # Number of such pairs is sum(v * (v - 1) // 2 for v in counts.values())
    
    # For s > t, we need P[t-1] ≡ P[s-1] - total_sum_mod (mod M)
    # Let r2 = P[s-1]. We need P[t-1] ≡ (r2 - total_sum_mod) % M.
    # For a fixed r2, there are counts[r2] choices for s and counts[(r2 - total_sum_mod) % M] choices for t.
    # However, we must ensure t < s. 
    # This is tricky without loops. Let's rethink.
    
    # Total pairs (s, t) such that distance is 0 mod M:
    # Let x = P[s-1] and y = P[t-1].
    # If s < t: y - x ≡ 0 mod M  => y ≡ x mod M
    # If s > t: (total - x) + y ≡ 0 mod M => y ≡ x - total mod M
    
    # Let's calculate the sum for all s, t in [1, N] where s != t:
    # For each s, we need t such that:
    # t > s and P[t-1] ≡ P[s-1] mod M
    # OR t < s and P[t-1] ≡ P[s-1] - total_sum_mod mod M
    
    # Let's use the property: 
    # Total = Sum_{r=0 to M-1} (counts[r] * counts[(r - total_sum_mod) % M])
    # But we must subtract cases where s = t (which is not allowed).
    # s = t implies P[s-1] ≡ P[s-1] - total_sum_mod mod M, 
    # which means total_sum_mod ≡ 0 mod M.
    # If total_sum_mod ≡ 0 mod M, then for each s, t is any other index with the same remainder.
    
    # Correct Logic:
    # For each s ∈ {1...N}, let r = P[s-1].
    # We need t ∈ {1...N}, t ≠ s such that:
    # 1. t > s and P[t-1] ≡ r (mod M)
    # 2. t < s and P[t-1] ≡ r - total_sum_mod (mod M)
    
    # Let's sum over all r:
    # For a fixed r, let C(r) = counts[r].
    # The number of pairs (s, t) with s < t and P[s-1] = P[t-1] = r is C(r)*(C(r)-1)//2.
    # The number of pairs (s, t) with s > t and P[t-1] = (r - total_sum_mod)%M and P[s-1] = r:
    # This is a bit like a convolution.
    # Let r_target = (r - total_sum_mod) % M.
    # If r_target == r, we have C(r)*(C(r)-1)//2 pairs.
    # If r_target != r, we have C(r) * C(r_target) pairs, but we only count t < s.
    
    # Actually, if we sum C(r) * C((r - total_sum_mod) % M) over all r,
    # we are counting pairs (s, t) such that P[t-1] ≡ P[s-1] - total_sum_mod (mod M).
    # If total_sum_mod ≡ 0 mod M, this is Sum C(r)^2. 
    # Since s != t, we subtract N, getting Sum C(r)^2 - N.
    # Since total_sum_mod ≡ 0, the condition s < t and s > t are the same.
    # The answer is (Sum C(r)^2 - N).
    
    # If total_sum_mod ≢ 0 mod M:
    # For each s, we need t such that:
    # (t > s AND P[t-1] ≡ P[s-1]) OR (t < s AND P[t-1] ≡ P[s-1] - total_sum_mod)
    # Let r = P[s-1] and r' = (r - total_sum_mod) % M.
    # We want count(t > s, P[t-1] == r) + count(t < s, P[t-1] == r').
    # Summing over s:
    # Sum_{s} [ (count(t > s, P[t-1] == r)) + (count(t < s, P[t-1] == r')) ]
    # = Sum_{r} [ C(r)*(C(r)-1)//2 + C(r)*C(r') ] 
    # Wait, the second term C(r)*C(r') counts all pairs (t, s) where P[t-1]=r' and P[s-1]=r.
    # Since r' != r, the condition t < s is naturally handled if we consider 
    # that for any pair of indices {i, j} with P[i]=r' and P[j]=r, 
    # exactly one of them is smaller.
    # So if r' != r, the number of pairs (s, t) is C(r) * C(r').
    # But we only care about the specific relation: t < s.
    # Let' same_rem = Sum_{r} C(r)*(C(r)-1)//2
    # Let diff_rem = Sum_{r} C(r) * C((r - total_sum_mod) % M) where r != (r - total_sum_mod) % M
    # The total is same_rem + (diff_rem if we only count t < s).
    # Actually, for any r, the number of pairs (s, t) with s < t and P[s-1]=P[t-1]=r is C(r)(C(r)-1)//2.
    # For any r, the number of pairs (s, t) with s > t and P[t-1]=(P[s-1]-total)%M is:
    # we need to count pairs (t, s) such that t < s and P[t-1] = (P[s-1] - total)%M.
    # Let r_t = P[t-1] and r_s = P[s-1]. We need r_t = (r_s - total) % M.
    # This is equivalent to r_s = (r_t + total) % M.
    # For a fixed t, any s > t with P[s-1] = (r_t + total) % M works.
    
    # Let',s be the indices where P is r, and t,s be indices where P is r'.
    # We want to count (s, t) such that:
    # (s < t and P[s-1] == P[t-1]) OR (s > t and P[t-1] == (P[s-1] - total) % M)
    
    # Let's use the property:
    # Sum_{s < t, P[s-1]=P[t-1]} 1 = Sum_{r} C(r)(C(r)-1)//2
    # Sum_{s > t, P[t-1]=(P[s-1]-total)%M} 1:
    # Let r_s = r. Then r_t = (r - total) % M.
    # We need to count pairs (t, s) such that t < s, P[t-1] = r_t, P[s-1] = r_s.
    # If r_t == r_s (i.e., total % M == 0), this is the same as the first case.
    # If r_t != r_s, then for any t with P[t-1]=r_t and s with P[s-1]=r_s, 
    # we don't know if t < s.
    # BUT, we can iterate through the array P once.
    # For each s, we need:
    # (count of r in P[s...N-1]) + (count of r_t in P[0...s-1])
    
    # Since we can't use loops, we use a list comprehension to calculate 
    # the "t < s" part by tracking the prefix counts.
    # However, we can't maintain state in a list comprehension.
    # We can use a different approach:
    # The total number of pairs (s, t) such that s > t and P[t-1] = r_t, P[s-1] = r_s
    # is simply the number of times r_t appears before r_s in the sequence P.
    
    # Let's use the fact that:
    # Total = Sum_{r} [ C(r)*(C(r)-1)//2 ]  <-- for s < t, P[s-1] == P[t-1]
    #       + Sum_{s, t: t < s} [ P[t-1] == (P[s-1] - total) % M ]
    
    # To calculate the second sum without loops:
    # For each r, let indices(r) be the list of indices where P[i] == r.
    # For a fixed r_s, we need to count how many t < s have P[t-1] = r_t.
    # This is Sum_{s: P[s-1]=r_s} (count of r_t in P[0...s-1]).
    
    # We can use a clever trick:
    # Sum_{t < s