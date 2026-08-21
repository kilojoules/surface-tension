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
    # We use accumulate to avoid loops.
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    # P now has N+1 elements. P[0] is 0, P[N] is the total sum % M.
    # The distance from rest area s to t (1 <= s, t <= N) is:
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == P[N] % M
    
    # We only care about P[0] to P[N-1] for the starting/ending points.
    # Let S = P[N] (the total sum % M).
    S = P[N]
    prefixes = P[:N]
    counts = Counter(prefixes)
    
    # For a fixed s, we need t such that:
    # 1. t > s and P[t-1] == P[s-1] (mod M)
    # 2. t < s and P[t-1] == (P[s-1] - S) (mod M)
    
    # Total pairs (s, t) with s < t and P[s-1] == P[t-1] is:
    # sum(count * (count - 1) // 2) for each unique value in prefixes.
    # However, the condition s < t and t < s is symmetric.
    # Let's use the property: 
    # For each value v in prefixes, there are count[v] indices.
    # Each pair (i, j) with i < j contributes 1 if P[i] == P[j].
    # Each pair (i, j) with i < j contributes 1 if (P[j] - P[i]) % M == S.
    
    # Let's refine:
    # We seek pairs (s, t) such that dist(s, t) % M == 0.
    # Let x = P[s-1] and y = P[t-1].
    # If s < t: (y - x) % M == 0  => y == x
    # If s > t: (S - x + y) % M == 0 => x - y == S (mod M) => y == (x - S) % M
    
    # Let freq be the Counter of P[0...N-1].
    # For each x in freq:
    #   Pairs (s, t) with s < t and P[s-1] == x and P[t-1] == x:
    #   This is handled by combinations.
    #   Pairs (s, t) with s > t and P[s-1] == x and P[t-1] == (x - S) % M:
    #   This is handled by freq[x] * freq[(x - S) % M].
    
    # Special case: If S == 0, then (x - S) % M == x.
    # The condition s < t and P[s-1] == P[t-1] AND s > t and P[s-1] == P[t-1]
    # simply means any two distinct indices i, j with P[i] == P[j] work.
    # That is freq[x] * (freq[x] - 1).
    
    # If S != 0:
    # Pairs are:
    # 1. s < t and P[s-1] == P[t-1]
    # 2. s > t and P[t-1] == (P[s-1] - S) % M
    
    # Let's use a different approach to avoid loops:
    # Total = sum(freq[x] * (freq[x] - 1) // 2)  <-- for s < t
    #       + sum(freq[x] * freq[(x - S) % M])   <-- for s > t
    # But wait, if S == 0, the second term becomes sum(freq[x] * freq[x]).
    # That would double count.
    
    # Correct Logic:
    # For every pair of indices i, j (0 <= i < j < N):
    # Pair (s=i+1, t=j+1) is valid if (P[j] - P[i]) % M == 0
    # Pair (s=j+1, t=i+1) is valid if (S - P[j] + P[i]) % M == 0
    
    # Let's calculate:
    # Part 1: sum(freq[x] * (freq[x] - 1) // 2) for all x
    # Part 2: sum(freq[x] * freq[(x - S) % M]) for all x, but only for s > t.
    # Actually, for a fixed pair i < j, 
    # (s, t) = (i+1, j+1) is valid if P[i] == P[j]
    # (s, t) = (j+1, i+1) is valid if P[i] == (P[j] - S) % M
    
    # Let's use the sum of freq[x] * freq[(x - S) % M] for all x.
    # If S == 0, this is sum(freq[x]^2). But we need s != t, so freq[x]*(freq[x]-1).
    # If S != 0, the two conditions (P[i] == P[j]) and (P[i] == (P[j] - S) % M)
    # are mutually exclusive for a fixed pair i, j.
    
    # Total = sum(freq[x] * (freq[x] - 1) // 2 for x in freq) # for s < t
    #       + sum(freq[x] * freq[(x - S) % M] for x in freq) # for s > t
    # Wait, the second term: for a fixed x, freq[x] is count of s, 
    # and freq[(x-S)%M] is count of t. Since we need s > t, 
    # we can't just multiply. We need to know how many t < s.
    
    # Let's reconsider:
    # Total = sum_{i < j} [P[i] == P[j]] + sum_{i < j} [P[i] == (P[j] - S) % M]
    # The first term is sum(freq[x] * (freq[x] - 1) // 2)
    # The second term: for each j, we need count of i < j such that P[i] == (P[j] - S) % M.
    # This is sum_{j=0 to N-1} (count of P[i] == (P[j] - S) % M for i < j).
    
    # To do this without loops, we can use the fact that:
    # sum_{i < j} [P[i] == V] = (freq[V] * (freq[V] - 1) // 2) if V is the same.
    # But here V depends on j.
    # Let's use: sum_{i < j} [P[i] == (P[j] - S) % M]
    # If S == 0, this is sum(freq[x] * (freq[x] - 1) // 2).
    # If S != 0, this is NOT simply freq[x] * freq[y].
    # Actually, it is. For any two indices i, j, one is smaller.
    # If P[i] == (P[j] - S) % M, then either (i < j and s=j+1, t=i+1) 
    # or (j < i and s=i+1, t=j+1).
    # Only one of these can be true for a fixed pair of indices.
    # So for S != 0, the number of pairs (s, t) with s > t is 
    # sum_{j} (count of i < j such that P[i] == (P[j] - S) % M).
    # This is equivalent to sum_{x} (count of i such that P[i] == x) * (count of j such that P[j] == (x + S) % M)
    # BUT only for i < j. This is tricky.
    
    # Let's use the property:
    # Total = sum_{i < j} ([P[i] == P[j]] + [P[i] == (P[j] - S) % M])
    # If S == 0: Total = sum(freq[x] * (freq[x] - 1))
    # If S != 0: 
    # The condition P[i] == P[j] and P[i] == (P[j] - S) % M cannot both be true.
    # The total number of pairs (i, j) with i != j such that (P[j] - P[i]) % M == S
    # is sum(freq[x] * freq[(x + S) % M]).
    # And the number of pairs (i, j) with i != j such that (P[j] - P[i]) % M == 0
    # is sum(freq[x] * (freq[x] - 1)).
    # Wait, the distance from s to t is:
    # If s < t: (P[t-1] - P[s-1]) % M
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M
    # We want these to be 0 mod M.
    # 1. s < t: P[t-1] == P[s-1] (mod M)
    # 2. s > t: P[s-1] - P[t-1] == P[N] (mod M)
    
    # Let freq be the Counter of P[0...N-1].
    # For s < t, the number of pairs is sum(freq[x] * (freq[x] - 1) // 2).
    # For s > t, the number of pairs is sum_{j < i} [P[i] - P[j] == S (mod M)].
    # This is sum_{j < i} [P[j] == (P[i] - S) (mod M)].
    
    # Let's use a different trick for s > t.
    # Let P' be the prefix sums. We want i > j such that P'[i] - P'[j] \equiv S \pmod M.
    # This is equivalent to sum_{i=0}^{N-1} (count of j < i such that P'[j] == (P'[i] - S) % M).
    # We can compute this by iterating through the list and keeping track of counts,
    # but we can't use loops. 
    # However, we can use the fact that:
    # sum_{i < j} [P[i] == (P[j] - S) % M] + sum_{i > j} [P[i] == (P[j] - S) % M] 
    # = sum_{x} freq[x] * freq[(x + S) % M] (excluding i=j).
    
    # Let's use the property:
    # For any two distinct indices i, j, one is smaller.
    # If P[i] == P[j], then the pair (s=i+1, t=j+1) is valid if i < j.
    # If (P[i] - P[j]) % M ==