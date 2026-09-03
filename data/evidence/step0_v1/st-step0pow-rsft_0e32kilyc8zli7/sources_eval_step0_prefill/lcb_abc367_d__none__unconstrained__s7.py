import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the distance from rest area 1 to rest area i.
    # P[1] = 0
    # P[2] = A[0]
    # P[3] = A[0] + A[1]
    # ...
    # P[N] = A[0] + ... + A[N-2]
    # Total distance L = A[0] + ... + A[N-1]
    
    # The distance from s to t (s < t) is P[t] - P[s].
    # The distance from s to t (s > t) is (L - P[s]) + P[t].
    
    # We want distance % M == 0.
    # For s < t: (P[t] - P[s]) % M == 0  => P[t] % M == P[s] % M.
    # For s > t: (L + P[t] - P[s]) % M == 0 => (P[s] - P[t]) % M == L % M.
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    L = P[N]
    L_mod = L % M
    
    # We only care about P[1] to P[N]
    # Let X_i = P[i] % M for i = 1 to N.
    X = [P[i] % M for i in range(1, N + 1)]
    
    # Count occurrences of each remainder
    count = {}
    for val in X:
        count[val] = count.get(val, 0) + 1
        
    ans = 0
    
    # Case 1: s < t
    # For each remainder r, if there are k people with that remainder,
    # there are k*(k-1)//2 pairs (s, t) with s < t.
    for r in count:
        k = count[r]
        ans += k * (k - 1) // 2
        
    # Case 2: s > t
    # We need (X[s-1] - X[t-1]) % M == L_mod
    # Which is X[s-1] % M == (L_mod + X[t-1]) % M
    # We iterate through all possible remainders r for X[t-1]
    for r in count:
        k_t = count[r]
        r_s = (L_mod + r) % M
        k_s = count.get(r_s, 0)
        
        # This counts pairs (s, t) where s > t.
        # However, we must ensure s != t.
        # If r_s == r, we are counting pairs where X[s-1] == X[t-1].
        # But the condition s > t is handled by the fact that we are 
        # looking for the number of pairs (s, t) such that s > t.
        # For a fixed t, any s > t with the required remainder works.
        # This is tricky to do with just the global count.
        
        # Let's rethink:
        # Total pairs (s, t) with s != t is N*(N-1).
        # A pair (s, t) is valid if:
        # 1. s < t and (P[t] - P[s]) % M == 0
        # 2. s > t and (L + P[t] - P[s]) % M == 0
        
        # Let's use the property:
        # For a fixed s, we want t != s such that:
        # If t > s: P[t] % M == P[s] % M
        # If t < s: P[t] % M == (P[s] - L) % M
        
        # Let's iterate over all s from 1 to N:
        # Let r_s = P[s] % M.
        # Number of t > s such that P[t] % M == r_s is (count[r_s] - (number of i <= s with P[i]%M == r_s))
        # Number of t < s such that P[t] % M == (r_s - L) % M is (number of i < s with P[i]%M == (r_s - L)%M)
        
    # To implement this efficiently:
    # We can just use the global counts and then subtract the cases where s=t.
    # For a fixed s, the number of t != s such that the condition holds is:
    # (count[r_s] - 1) if (r_s - L) % M != r_s
    # (count[r_s] - 1) + (count[(r_s - L) % M]) if (r_s - L) % M != r_s
    # Wait, that's not quite right.
    
    # Let's use the logic:
    # For each s, we want t such that:
    # If t > s, P[t] % M = P[s] % M
    # If t < s, P[t] % M = (P[s] - L) % M
    
    # Let's use a running count for t < s and a total count for t > s.
    # total_count[r] is the number of i in {1...N} such that P[i] % M == r.
    # current_count[r] is the number of i in {1...s-1} such that P[i] % M == r.
    
    # For s = 1 to N:
    # r_s = P[s] % M
    # t > s: total_count[r_s] - (current_count[r_s] + 1)
    # t < s: current_count[(r_s - L_mod) % M]
    
    # Let's re-calculate ans using this logic.
    ans = 0
    current_count = {}
    for s in range(1, N + 1):
        r_s = P[s] % M
        
        # t > s
        t_gt_s = count.get(r_s, 0) - (current_count.get(r_s, 0) + 1)
        # t < s
        r_t = (r_s - L_mod) % M
        t_lt_s = current_count.get(r_t, 0)
        
        ans += t_gt_s + t_lt_s
        
        current_count[r_s] = current_count.get(r_s, 0) + 1
        
    print(ans)

if __name__ == "__main__":
    solve()