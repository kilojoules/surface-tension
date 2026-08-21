import sys

def solve():
    # Read N and M from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, m = map(int, line1)
        
        # Read A_i values
        a = []
        while len(a) < n:
            a.extend(map(int, sys.stdin.readline().split()))
    except ValueError:
        return

    # Let P[i] be the clockwise distance from rest area 1 to rest area i.
    # P[1] = 0
    # P[2] = A_1
    # P[3] = A_1 + A_2
    # ...
    # P[N] = A_1 + ... + A_{N-1}
    # Total distance L = A_1 + ... + A_N
    
    p = [0] * (n + 1)
    for i in range(n):
        p[i+1] = p[i] + a[i]
    
    total_dist = p[n]
    
    # The distance from s to t (s != t) clockwise is:
    # If s < t: dist = P[t] - P[s]
    # If s > t: dist = (total_dist - P[s]) + P[t]
    
    # We want dist % M == 0.
    # Case 1: s < t
    # P[t] % M == P[s] % M
    # Case 2: s > t
    # (total_dist + P[t] - P[s]) % M == 0  =>  P[s] % M == (total_dist + P[t]) % M
    
    # Let's count occurrences of each remainder P[i] % M for i = 1 to N.
    # Note: P[1] is the distance to rest area 1, P[2] to 2, ..., P[N] to N.
    counts = [0] * m
    for i in range(1, n + 1):
        counts[p[i-1] % m] += 1
    
    # Note: The logic above uses P[0]...P[N-1] because indices in Python are 0-based.
    # Let X_i = P[i-1] % M for i = 1...N.
    # We want (X_t - X_s) % M == 0 for s < t
    # We want (total_dist + X_t - X_s) % M == 0 for s > t
    
    # Let's redefine: 
    # Let S be the set of remainders {P[0]%M, P[1]%M, ..., P[N-1]%M}
    # Total pairs (s, t) with s != t is N*(N-1).
    # For a fixed pair (s, t) with s < t:
    # Valid if P[t-1] % M == P[s-1] % M
    # For a fixed pair (s, t) with s > t:
    # Valid if P[s-1] % M == (total_dist + P[t-1]) % M
    
    # Let's use a frequency map for P[i-1] % M
    freq = [0] * m
    for i in range(n):
        freq[p[i] % m] += 1
        
    ans = 0
    
    # For s < t:
    # For each remainder r, there are freq[r] elements.
    # The number of pairs (s, t) with s < t and P[s-1]%M == P[t-1]%M is freq[r] * (freq[r] - 1) // 2.
    # However, this is only for s < t. The problem asks for pairs (s, t).
    # Let's iterate through all possible remainders r.
    # For a fixed r, let the indices be i_1 < i_2 < ... < i_k.
    # Pairs (i_j, i_l) with j < l are valid. Total: k*(k-1)//2.
    
    # For s > t:
    # We need P[s-1] % M == (total_dist + P[t-1]) % M.
    # Let total_rem = total_dist % M.
    # We need P[s-1] % M == (total_rem + P[t-1]) % M.
    # For a fixed t, we need to count s > t such that P[s-1] % M == (total_rem + P[t-1]) % M.
    
    # To solve this efficiently:
    # For s < t:
    # We can use the freq array: sum(freq[r] * (freq[r]-1) // 2)
    # But this only covers s < t.
    # For s > t:
    # We need to count pairs (s, t) such that s > t and P[s-1] % M == (total_rem + P[t-1]) % M.
    # Let's iterate through the array P[0...N-1] and keep track of counts of remainders seen so far.
    
    # Let's use a simpler approach:
    # Total pairs (s, t) where s != t:
    # 1. s < t: P[t-1] % M == P[s-1] % M
    # 2. s > t: P[s-1] % M == (total_dist + P[t-1]) % M
    
    # Part 1: s < t
    # For each r in 0...M-1, if freq[r] = k, there are k*(k-1)//2 pairs.
    ans_s_lt_t = 0
    for k in freq:
        ans_s_lt_t += k * (k - 1) // 2
        
    # Part 2: s > t
    # We need P[s-1] % M == (total_dist + P[t-1]) % M where s > t.
    # Let total_rem = total_dist % M.
    # We iterate i from 0 to N-1. At step i, P[i] is P[s-1].
    # We want to count j < i such that P[j] % M == (P[i] - total_rem) % M.
    
    ans_s_gt_t = 0
    current_freq = [0] * m
    total_rem = total_dist % m
    for i in range(n):
        # P[i] is the remainder for s
        # We need P[j] % M == (P[i] - total_rem) % M for j < i
        target = (p[i] - total_rem) % m
        ans_s_gt_t += current_freq[target]
        current_freq[p[i] % m] += 1
        
    print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()