```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of rest areas, M: the divisor
    n = int(input_data[0])
    m = int(input_data[1])
    a = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M
    # p[i] = (A_1 + ... + A_{i-1}) % M
    # The distance from s to t (s < t) is (p[t-1] - p[s-1]) % M
    # The distance from s to t (s > t) is (TotalSum - (p[s-1] - p[t-1])) % M
    
    # We use accumulate to get prefix sums and map them to modulo M
    # p will have length N+1: [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    p = list(map(lambda x: x % m, accumulate(a, initial=0)))
    
    # Let S be the total sum of A_i modulo M
    s_total = p[n]
    
    # We want to find pairs (s, t) such that dist(s, t) % M == 0
    # Case 1: s < t
    # (p[t-1] - p[s-1]) % M == 0  => p[t-1] == p[s-1]
    # Case 2: s > t
    # (s_total - (p[s-1] - p[t-1])) % M == 0 => p[s-1] - p[t-1] == s_total (mod M)
    # => p[t-1] == (p[s-1] - s_total) % M
    
    # Count occurrences of each prefix sum modulo M
    # We only consider p[0] to p[n-1] because the rest areas are 1 to N
    counts = Counter(p[:n])
    
    # For a fixed s, we need to count t such that:
    # 1. t > s and p[t-1] == p[s-1]
    # 2. t < s and p[t-1] == (p[s-1] - s_total) % m
    
    # However, it's easier to use combinatorics:
    # Total pairs = Sum_{for each val v} (count[v] * (count[v]-1) // 2) 
    # This covers s < t where p[s-1] == p[t-1].
    # For s > t, we need p[s-1] - p[t-1] == s_total (mod M).
    # This is equivalent to p[t-1] == (p[s-1] - s_total) % M.
    
    # Let's use the property:
    # For every pair {i, j} with i < j:
    # Clockwise i -> j is (p[j] - p[i]) % M
    # Clockwise j -> i is (s_total - (p[j] - p[i])) % M
    
    # We want (p[j] - p[i]) % M == 0 OR (s_total - (p[j] - p[i])) % M == 0
    # Note: if s_total % M == 0, then (p[j] - p[i]) % M == 0 implies both are 0.
    # But the problem says s != t.
    
    # Let x = p[i] and y = p[j].
    # Pair is valid if y - x \equiv 0 (mod M) OR y - x \equiv s_total (mod M).
    
    # If s_total % M == 0:
    # We only need y == x. Number of pairs is sum(c * (c-1) // 2) for each count c.
    # Since we can go s -> t or t -> s, and both are 0 mod M, we multiply by 2?
    # No, the question asks for pairs (s, t).
    # If p[i] == p[j], then dist(i, j) = 0 mod M AND dist(j, i) = 0 mod M.
    # That's 2 pairs for every combination of 2 indices.
    
    # If s_total % M != 0:
    # We need y - x \equiv 0 (mod M) OR y - x \equiv s_total (mod M).
    # These two conditions are mutually exclusive.
    # For y == x: we get 2 * (c*(c-1)//2) pairs? No.
    # If p[i] == p[j] (i < j), then dist(i, j) = 0 mod M.
    # If p[j] - p[i] == s_total mod M, then dist(j, i) = 0 mod M.
    
    # Correct Logic:
    # For each i \in {0, ..., N-1}, we look for j \in {0, ..., N-1}, j != i.
    # dist(i+1, j+1) is a multiple of M if:
    # 1. i < j and (p[j] - p[i]) % M == 0
    # 2. i > j and (s_total - (p[i] - p[j])) % M == 0
    
    # Let's re-evaluate:
    # Pair (s, t) is valid if:
    # s < t: p[t-1] - p[s-1] \equiv 0 (mod M)  => p[t-1] \equiv p[s-1] (mod M)
    # s > t: p[s-1] - p[t-1] \equiv s_total (mod M) => p[t-1] \equiv p[s-1] - s_total (mod M)
    
    # Let C(v) be the count of prefix sums equal to v.
    # For a fixed s, the number of t's is:
    # (count of p[j] == p[s-1] for j > s-1) + (count of p[j] == (p[s-1] - s_total) % M for j < s-1)
    
    # This is equivalent to:
    # Sum_{v=0 to M-1} [ C(v) * (C(v)-1) // 2 ]  <-- this is s < t and p[s-1] == p[t-1]
    # PLUS
    # Sum_{v=0 to M-1} [ C(v) * C((v - s_total) % M) ] <-- this is s > t and p[t-1] == (p[s-1] - s_total) % M
    # BUT we must be careful if s_total % M == 0.
    
    # If s_total % M == 0:
    # s < t: p[s-1] == p[t-1]
    # s > t: p[t-1] == p[s-1]
    # Total = 2 * Sum(C(v) * (C(v)-1) // 2) = Sum(C(v) * (C(v)-1))
    
    # If s_total % M != 0:
    # s < t: p[s-1] == p[t-1]
    # s > t: p[t-1] == (p[s-1] - s_total) % M
    # Total = Sum(C(v) * (C(v)-1) // 2) + Sum(C(v) * C((v - s_total) % M))
    # Wait, the second term is Sum_{v} C(v) * C(v_prev). 
    # For every pair {i, j} with i < j, we check if p[i] == p[j] (then s=i+1, t=j+1) 
    # and if p[j] - p[i] == s_total (mod M) (then s=j+1, t=i+1).
    # These are distinct conditions if s_total % M != 0.
    
    # Let's use the most robust way:
    # For every pair i, j with 0 <= i < j < N:
    # Pair (i+1, j+1) is valid if p[j] - p[i] \equiv 0 (mod M)
    # Pair (j+1, i+1) is valid if s_total - (p[j] - p[i]) \equiv 0 (mod M)
    
    # Count of (i, j) such that p[j] - p[i] \equiv 0 (mod M) is Sum(C(v)*(C(v)-1)//2)
    # Count of (i, j) such that p[j] - p[i] \equiv s_total (mod M) is:
    # We need p[j] - p[i] \equiv s_total (mod M).
    # This is Sum_{v} (count of p[i] == v) * (count of p[j] == (v + s_total) % M)
    # But we need i < j.
    
    # Actually, we can just use:
    # For each v, C(v) is the number of times v appears in p[0...N-1].
    # The number of pairs (s, t) with s < t such that dist(s, t) % M == 0 is Sum(C(v)*(C(v)-1)//2).
    # The number of pairs (s, t) with s > t such that dist(s, t) % M == 0 is:
    # dist(s, t) = (s_total - (p[s-1] - p[t-1])) % M.
    # We want s_total - p[s-1] + p[t-1] \equiv 0 (mod M) => p[t-1] \equiv p[s-1] - s_total (mod M).
    # Let v = p[s-1]. We need p[t-1] = (v - s_total) % M for t < s.
    # This is a convolution-like problem but we can solve it by iterating through the array.
    
    # Let's use a different approach:
    # For every pair of indices (i, j) with 0 <= i < j < N:
    # We have two potential pairs: (i+1, j+1) and (j+1, i+1).
    # (i+1, j+1) is valid if p[j] - p[i] \equiv 0 (mod M)
    # (j+1, i+1) is valid if p[j] - p[i] \equiv s_total (mod M)
    
    # Let C(v) be the total count of value v in p[0...N-1].
    # The number of pairs (i, j) with i < j such that p[j] - p[i] \equiv 0 (mod M) is Sum(C(v)*(C(v)-1)//2).
    # The number of pairs (i, j) with i < j such that p[j] - p[i] \equiv s_total (mod M) is:
    # This is tricky because of the i < j constraint.
    # Let's use the property: 
    # Total pairs (s, t) = (pairs with s < t) + (pairs with s > t)
    # s < t: p[t-1] - p[s-1] \equiv 0 (mod M)
    # s > t: p[s-1] - p[t-1] \equiv s_total (mod M)
    
    # Let's process the array p and maintain counts of values seen so far.
    # For each element p[j] (where j is the index of the second rest area in the pair):
    # If we treat j as 't' (s < t), we need p[s-1] == p[j].
    # If we treat j as 's' (s > t), we need p[t-1] == (p[j] - s_total) % M.
    
    # We can use a list comprehension to calculate this without explicit loops.
    # But we need the state of counts at each step.
    # We can use a trick: 
    # Total = Sum_{v} (C(v)*(C(v)-1)//2) + Sum_{v} (count of pairs (