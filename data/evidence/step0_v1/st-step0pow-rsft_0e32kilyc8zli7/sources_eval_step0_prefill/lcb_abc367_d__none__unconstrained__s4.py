import sys

def solve():
    # Read N and M from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A, where P_0 = 0.
    # P_i = A_1 + ... + A_i.
    # The distance from rest area s to t (s < t) is P_{t-1} - P_{s-1}.
    # The distance from rest area s to t (s > t) is (P_N - P_{s-1}) + P_{t-1}.
    # We want distance % M == 0.
    
    # Let S_i = P_i % M.
    # For s < t: (S_{t-1} - S_{s-1}) % M == 0  => S_{t-1} == S_{s-1}
    # For s > t: (S_N - S_{s-1} + S_{t-1}) % M == 0 => S_{s-1} - S_{t-1} == S_N % M
    
    # Let's redefine indices to be 0-indexed for convenience.
    # Rest areas are 0, 1, ..., N-1.
    # Distance from s to t (s < t) is sum(A[s...t-1]).
    # Distance from s to t (s > t) is sum(A[s...N-1]) + sum(A[0...t-1]).
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = (P[i] + A[i]) % M
        
    # S_N is P[N]
    SN = P[N]
    
    # We are looking for pairs (s, t) with 0 <= s, t < N and s != t.
    # Case 1: s < t
    # Distance is (P[t] - P[s]) % M == 0  => P[t] == P[s]
    # Case 2: s > t
    # Distance is (P[N] - P[s] + P[t]) % M == 0 => P[s] - P[t] == SN % M
    
    # Count occurrences of each remainder in P[0...N-1]
    count = {}
    for i in range(N):
        val = P[i]
        count[val] = count.get(val, 0) + 1
        
    ans = 0
    
    # For Case 1 (s < t), for each remainder r, if it appears k times,
    # there are k*(k-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # Let's iterate through all possible remainders r.
    for r in count:
        k = count[r]
        ans += k * (k - 1) // 2
        
    # For Case 2 (s > t), we need P[s] - P[t] \equiv SN (mod M).
    # This is P[s] \equiv P[t] + SN (mod M).
    # For each t, we need to count s > t such that P[s] \equiv P[t] + SN (mod M).
    # Actually, it's easier to just use the total counts.
    # For a fixed t, any s such that P[s] \equiv P[t] + SN (mod M) and s > t works.
    # But the condition s > t is already handled by the "Case 1" logic if we just
    # look at all pairs and subtract.
    
    # Let's rethink:
    # We want pairs (s, t) such that:
    # If s < t: P[t] \equiv P[s] (mod M)
    # If s > t: P[s] \equiv P[t] + SN (mod M)
    
    # Let's use the counts of P[0...N-1].
    # For Case 1: s < t and P[s] == P[t].
    # This is simply sum( count[r] * (count[r]-1) // 2 )
    
    # For Case 2: s > t and P[s] == (P[t] + SN) % M.
    # Let r_t = P[t] and r_s = (r_t + SN) % M.
    # We want to count pairs (s, t) such that s > t and P[s] == r_s.
    # This is harder to do with just total counts because of the s > t constraint.
    # Let's use a running count.
    
    # Total pairs (s, t) is the sum of:
    # 1. Pairs (s, t) with s < t and P[s] == P[t]
    # 2. Pairs (s, t) with s > t and P[s] == (P[t] + SN) % M
    
    # Part 1:
    res1 = 0
    for r in count:
        k = count[r]
        res1 += k * (k - 1) // 2
        
    # Part 2:
    # We need s > t and P[s] == (P[t] + SN) % M.
    # Iterate s from 0 to N-1, and maintain counts of P[t] seen so far.
    res2 = 0
    t_counts = {}
    for s in range(N):
        # We need P[t] such that P[s] == (P[t] + SN) % M
        # P[t] == (P[s] - SN) % M
        target = (P[s] - SN) % M
        res2 += t_counts.get(target, 0)
        
        # Update t_counts for the next s
        val_t = P[s]
        t_counts[val_t] = t_counts.get(val_t, 0) + 1
        
    print(res1 + res2)

if __name__ == "__main__":
    solve()