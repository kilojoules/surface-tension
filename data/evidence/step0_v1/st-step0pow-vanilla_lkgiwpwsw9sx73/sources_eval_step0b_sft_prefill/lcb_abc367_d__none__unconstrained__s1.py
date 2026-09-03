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
    
    # Calculate the prefix sums of A modulo M.
    # P[i] = (A_1 + ... + A_{i-1}) % M.
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M.
    # The distance clockwise from s to t is:
    # If s < t: (P[t-1] - P[s-1]) % M
    # If s > t: (P[N] - P[s-1] + P[t-1]) % M
    # In both cases, the distance is a multiple of M if P[s-1] == P[t-1] (mod M).
    # Note: s != t is required.
    
    # Use accumulate to get prefix sums and map them to their values modulo M.
    # We prepend 0 to handle the case where s=1.
    P = list(map(lambda x: x % M, accumulate([0] + A)))
    
    # We are looking for pairs (s, t) such that 1 <= s, t <= N and s != t,
    # and the clockwise distance from s to t is 0 mod M.
    # Let x = P[s-1] and y = P[t-1].
    # The distance is (y - x) mod M if s < t, and (P[N] - x + y) mod M if s > t.
    
    # Let S = P[N].
    # Condition 1: s < t and (y - x) % M == 0  => x == y
    # Condition 2: s > t and (S - x + y) % M == 0 => x - y == S % M
    
    S = P[N]
    
    # Count occurrences of each remainder in P[0...N-1]
    # P[N] is the total sum, we only care about the starting/ending points 1...N
    counts = Counter(P[:N])
    
    # For a fixed remainder 'v', the number of pairs (s, t) with s < t and P[s-1] == P[t-1] == v
    # is combinations(counts[v], 2).
    # Total for Condition 1: sum(c * (c - 1) // 2 for c in counts.values())
    ans_cond1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For Condition 2: s > t and (x - y) % M == S % M.
    # This is equivalent to x - y = S or x - y = S - M (since 0 <= x, y < M).
    # We need to count pairs (s, t) with t < s such that P[s-1] - P[t-1] \equiv S (mod M).
    # This is equivalent to P[t-1] \equiv P[s-1] - S (mod M).
    
    # For each s from 2 to N, we need the count of t < s such that P[t-1] == (P[s-1] - S) % M.
    # We can use a running counter to track the occurrences of P[0...s-2].
    
    # To avoid loops, we can use the fact that for a fixed v, 
    # we want to count pairs (s, t) with t < s such that P[t-1] = v and P[s-1] = (v + S) % M.
    # If v == (v + S) % M (which happens if S % M == 0), this is the same as Condition 1.
    # If v != (v + S) % M, then for every t with P[t-1] = v and every s with P[s-1] = (v + S) % M,
    # we can't immediately say t < s.
    
    # Correct logic:
    # Let v1 = P[t-1] and v2 = P[s-1].
    # We want (s, t) such that:
    # 1. s < t and v1 == v2
    # 2. s > t and (v2 - v1) % M == S % M
    
    # Let's use the property:
    # Total pairs = (Number of pairs (s, t) with s != t such that dist(s, t) % M == 0)
    # For a fixed s, we want t != s such that:
    # If t > s, P[t-1] = P[s-1]
    # If t < s, P[t-1] = (P[s-1] - S) % M
    
    # Let count(v) be the number of times v appears in P[0...N-1].
    # For a fixed s, the number of t's is:
    # (count(P[s-1]) - 1) [these are all t != s where P[t-1] == P[s-1]]
    # BUT we must split them into t > s and t < s.
    # t > s and P[t-1] == P[s-1]
    # t < s and P[t-1] == (P[s-1] - S) % M
    
    # Let's re-evaluate:
    # For each v \in {0, ..., M-1}, let C_v be the set of indices i \in {0, ..., N-1} such that P[i] = v.
    # We want to count (i, j) such that:
    # 1. i < j and P[i] == P[j]
    # 2. i > j and (P[i] - P[j]) % M == S % M
    
    # Part 1: sum(len(C_v) * (len(C_v) - 1) // 2)
    # Part 2: count (i, j) with j < i such that P[j] == (P[i] - S) % M.
    # This is sum_{i=1 to N-1} (count of P[j] == (P[i] - S) % M for j < i).
    
    # To calculate Part 2 without loops:
    # For a fixed v, let v_prev = (v - S) % M.
    # We want to count pairs (j, i) with j < i such that P[j] = v_prev and P[i] = v.
    # This is tricky because the relative order matters.
    
    # Let's use the property:
    # Total pairs = (Pairs with P[i] == P[j]) + (Pairs with P[i] - P[j] == S % M) - (Pairs with P[i] == P[j] AND S % M == 0)
    # Wait, if S % M == 0, then P[i] == P[j] is the only condition.
    # If S % M != 0, then P[i] == P[j] and (P[i] - P[j]) % M == S % M are mutually exclusive.
    
    # If S % M == 0:
    # Distance is (P[t-1] - P[s-1]) % M.
    # This is 0 if P[t-1] == P[s-1].
    # For each v, we have count(v) choices for s and count(v)-1 choices for t.
    # Total = sum(count(v) * (count(v) - 1))
    
    # If S % M != 0:
    # Condition 1: s < t and P[s-1] == P[t-1]
    # Condition 2: s > t and P[s-1] - P[t-1] == S % M (or S % M - M)
    # Let v = P[s-1]. Condition 1 is P[t-1] = v (t > s). Condition 2 is P[t-1] = (v - S) % M (t < s).
    # For a fixed v, let the indices be i_1 < i_2 < ... < i_k.
    # Condition 1 gives k(k-1)//2 pairs.
    # For Condition 2, we need P[t-1] = v_prev where v_prev = (v - S) % M.
    # Let the indices for v_prev be j_1 < j_2 < ... < j_m.
    # We need to count (j, i) such that j < i, P[j] = v_prev, P[i] = v.
    
    # This can be solved by iterating through the array once or using a different combinatorial approach.
    # Actually, we can just use the property:
    # For any two indices i, j with i < j:
    # The clockwise distance from i+1 to j+1 is (P[j] - P[i]) % M.
    # The clockwise distance from j+1 to i+1 is (P[N] - P[j] + P[i]) % M.
    # We want (P[j] - P[i]) % M == 0 OR (P[N] - P[j] + P[i]) % M == 0.
    
    # If P[j] == P[i], both are 0 mod M. But we only count the pair (s, t) once.
    # If P[j] == P[i], then s=i+1, t=j+1 is a pair, and s=j+1, t=i+1 is a pair.
    # If P[j] != P[i], then at most one of them can be 0 mod M.
    # (P[j] - P[i]) % M == 0  => P[j] == P[i]
    # (S - (P[j] - P[i])) % M == 0 => P[j] - P[i] == S % M => P[j] == (P[i] + S) % M
    
    # Let v = P[i] and w = P[j].
    # If v == w, we get 2 pairs: (i+1, j+1) and (j+1, i+1).
    # If w == (v + S) % M and v != w, we get 1 pair: (i+1, j+1).
    # If v == (w + S) % M and v != w, we get 1 pair: (j+1, i+1).
    
    # Total = sum_{v} (count(v) * (count(v) - 1)) + sum_{v != (v+S)%M} (count(v) * count((v+S)%M))
    # Note: the second term is for pairs (i, j) with i < j. 
    # Actually, if we just look at all pairs of indices {i, j}, the distance is 0 mod M if:
    # P[i] == P[j] (two pairs: (i+1, j+1) and (j+1, i+1))
    # OR P[j] - P[i] == S % M (one pair: (i+1, j+1))
    # OR P[i] - P[j] == S % M (one pair: (j+1, i+1))
    
    # If S % M == 0, then P[i] == P[j] is the only way. Ans = sum(count(v) * (count(v) - 1))
    # If S % M != 0, then P[i] == P[j] and P[j]-P[i] == S % M are disjoint.
    # Ans = sum(count(v) * (count(v) - 1)) + sum(count(v) * count((v + S) % M))
    
    S_mod = S % M
    if S_mod == 0:
        print(sum(c * (c - 1) for c in counts.values()))
    else:
        term1 = sum(c * (c - 1) for c in counts.values())
        term2 = sum(counts[v] * counts.get((v + S_mod) % M, 0) for v in counts)
        print(term1 + term2)

if __name__ == "__main__":
    solve()