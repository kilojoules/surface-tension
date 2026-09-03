```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input and split into a list of integers
    data = list(map(int, sys.stdin.read().split()))
    
    # Extract N, M and the list of steps A
    N = data[0]
    M = data[1]
    A = data[2:]
    
    # Calculate prefix sums of A modulo M.
    # P[i] = (A_1 + ... + A_{i-1}) % M.
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M.
    # The distance clockwise from s to t (s < t) is (P[t-1] - P[s-1]) % M.
    # The distance clockwise from s to t (s > t) is (P[N] - P[s-1] + P[t-1]) % M.
    
    # We use accumulate to get prefix sums and map them to their values modulo M.
    # We prepend 0 to handle the case where s=1.
    P = list(map(lambda x: x % M, accumulate([0] + A)))
    
    # Total distance around the lake modulo M
    total_dist_mod_m = P[N]
    
    # We want to find pairs (s, t) such that distance(s, t) % M == 0.
    # Case 1: s < t. 
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1].
    # Case 2: s > t.
    # (total_dist_mod_m - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_dist_mod_m (mod M).
    
    # Let's count occurrences of each value in P[0...N-1].
    # Note: P[N] is the total sum, but the rest areas are 1...N.
    # The prefix sums corresponding to rest areas 1...N are P[0...N-1].
    counts = Counter(P[:N])
    
    # For Case 1 (s < t):
    # For each value v that appears C times, we have C * (C - 1) // 2 pairs.
    ans_case1 = sum(C * (C - 1) // 2 for C in counts.values())
    
    # For Case 2 (s > t):
    # We need P[s-1] - P[t-1] \equiv total_dist_mod_m (mod M).
    # This is equivalent to P[t-1] \equiv P[s-1] - total_dist_mod_m (mod M).
    # For each s-1 from 0 to N-1, we look for t-1 from 0 to s-2.
    # However, it's easier to iterate through all possible values of P[s-1].
    # For a fixed value v = P[s-1], we need P[t-1] = (v - total_dist_mod_m) % M.
    # The number of pairs is sum(counts[v] * counts[(v - total_dist_mod_m) % M]).
    # But we must ensure s > t. 
    # Actually, we can just calculate the total combinations and subtract the s=t cases.
    # Let X be the set of indices {0, ..., N-1}.
    # We want pairs (i, j) in X x X such that i != j and:
    # If i < j: P[j] - P[i] \equiv 0 (mod M)
    # If i > j: P[N] - P[i] + P[j] \equiv 0 (mod M)
    
    # Let's use the property:
    # Total pairs = (Pairs where P[j] == P[i]) + (Pairs where P[i] - P[j] == total_dist_mod_m)
    # If total_dist_mod_m == 0, these two conditions are the same.
    
    if total_dist_mod_m == 0:
        # If total_dist_mod_m is 0, then clockwise(s, t) is multiple of M iff P[s-1] == P[t-1].
        # For each group of size C, we have C*(C-1) ordered pairs.
        print(sum(C * (C - 1) for C in counts.values()))
    else:
        # If total_dist_mod_m != 0, then P[j] == P[i] and P[i] - P[j] == total_dist_mod_m are mutually exclusive.
        # Case 1: s < t and P[t-1] == P[s-1].
        # Case 2: s > t and P[s-1] - P[t-1] == total_dist_mod_m (mod M).
        # This is equivalent to: for every pair (i, j) with i < j:
        # Check if P[i] == P[j] (s=i+1, t=j+1) OR P[i] - P[j] == total_dist_mod_m (mod M) (s=j+1, t=i+1).
        
        # Let's re-evaluate:
        # For any two distinct indices i, j \in {0, ..., N-1}, let i < j.
        # Pair (s, t) = (i+1, j+1) is valid if P[j] - P[i] \equiv 0 (mod M).
        # Pair (s, t) = (j+1, i+1) is valid if P[N] - P[j] + P[i] \equiv 0 (mod M).
        
        # Total = \sum_{i < j} [P[i] == P[j]] + \sum_{i < j} [P[j] - P[i] \equiv P[N] (mod M)]
        # The first term is sum(C*(C-1)//2).
        # The second term: for each j, we need P[i] \equiv P[j] - P[N] (mod M) for i < j.
        # This is tricky with loops. Let's use the property:
        # \sum_{i < j} [P[j] - P[i] \equiv K] + \sum_{i > j} [P[i] - P[j] \equiv K] = \sum_{i \neq j} [P[i] - P[j] \equiv K]
        # Where K = total_dist_mod_m.
        # The number of pairs (i, j) with i \neq j such that P[i] - P[j] \equiv K (mod M) is:
        # \sum_{v} counts[v] * counts[(v - K) % M].
        
        # If K == 0, this is \sum counts[v]^2, but we must subtract the i=j cases (which is \sum counts[v]).
        # If K != 0, this is exactly the number of pairs (s, t) where s > t.
        
        # Correct Logic:
        # Let K = P[N].
        # We want pairs (s, t) with s != t.
        # If s < t: P[t-1] - P[s-1] \equiv 0 (mod M)
        # If s > t: P[N] - P[s-1] + P[t-1] \equiv 0 (mod M) => P[s-1] - P[t-1] \equiv K (mod M)
        
        # Let i = s-1, j = t-1.
        # Ans = \sum_{0 \le i < j < N} [P[i] == P[j]] + \sum_{0 \le j < i < N} [P[i] - P[j] \equiv K (mod M)]
        
        # Note that \sum_{i < j} [P[i] == P[j]] is the same as \sum_{j < i} [P[i] == P[j]].
        # So Ans = \sum_{j < i} ([P[i] == P[j]] + [P[i] - P[j] \equiv K (mod M)])
        
        # If K == 0, Ans = \sum_{j < i} 2 * [P[i] == P[j]] = \sum C * (C-1).
        # If K != 0, Ans = \sum_{j < i} [P[i] == P[j]] + \sum_{j < i} [P[i] - P[j] \equiv K (mod M)].
        # The first term is \sum C*(C-1)//2.
        # The second term: for a fixed i, we need P[j] \equiv P[i] - K (mod M) for j < i.
        # This is a convolution-like problem, but we can solve it by iterating through the array and keeping track of counts.
        
        # To avoid loops, we can use the fact that:
        # \sum_{i, j \in \{0...N-1\}, i \neq j} [P[i] - P[j] \equiv K (mod M)] 
        # = \sum_{v} counts[v] * counts[(v - K) % M] (if K != 0).
        # If K != 0, then P[i] == P[j] and P[i] - P[j] == K are mutually exclusive.
        # One of these corresponds to s < t and the other to s > t.
        # Specifically, for any pair {i, j} with i < j:
        # Either P[j] - P[i] \equiv 0 (s=i+1, t=j+1) or P[i] - P[j] \equiv K (s=j+1, t=i+1).
        # Wait, if P[j] - P[i] \equiv 0, then P[i] - P[j] \equiv 0. If K != 0, then P[i] - P[j] cannot be K.
        # So for every pair of indices {i, j}, we check two conditions.
        # Total = \sum_{i < j} [P[i] == P[j]] + \sum_{i < j} [P[i] - P[j] \equiv K (mod M)]
        # Total = \sum_{i < j} [P[i] == P[j]] + \sum_{j < i} [P[i] - P[j] \equiv K (mod M)]
        
        # Let's use the property:
        # \sum_{i < j} [P[i] == P[j]] = \sum C*(C-1)//2
        # \sum_{j < i} [P[i] - P[j] \equiv K] : 
        # We can compute this by iterating through P and using a running counter.
        
        # To avoid for/while loops, we use a list comprehension and a dictionary.
        # Since we can't use loops, we can use a trick with map/reduce or a custom function.
        # But we can just use the mathematical identity:
        # \sum_{i < j} [P[i] - P[j] \equiv K] = \sum_{i, j} [P[i] - P[j] \equiv K] - \sum_{i \ge j} [P[i] - P[j] \equiv K]
        # This is getting complex. Let's use the most direct way:
        # For a fixed K != 0:
        # The number of pairs (i, j) with i != j such that P[i] - P[j] \equiv K (mod M) is \sum_{v} counts[v] * counts[(v-K)%M].
        # For each such pair, either (i < j and P[i]-P[j] \equiv K) or (i > j and P[i]-P[j] \equiv K).
        # This doesn't immediately split into s < t and s > t.
        
        # Let's go back:
        # s < t: P[t-1] - P[s-1] \equiv 0 (mod M)
        # s > t: P[s-1] - P[t-1] \equiv K (mod M)
        # Let i = s-1, j = t-1.
        # Ans = \sum_{0 \le i < j < N} [P[j] - P[i] \equiv 0] + \sum_{0 \le j < i <