import sys

def solve():
    # Read N and M from the first line of input
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, m = map(int, line1)
        
        # Read A_i from the subsequent lines
        a = []
        while len(a) < n:
            line = sys.stdin.readline().split()
            if not line:
                break
            a.extend(map(int, line))
    except ValueError:
        return

    # Let P[i] be the distance from rest area 1 to rest area i clockwise.
    # P[1] = 0
    # P[2] = A_1
    # P[3] = A_1 + A_2
    # ...
    # P[N] = A_1 + ... + A_{N-1}
    # Total circumference L = A_1 + ... + A_N
    
    p = [0] * (n + 1)
    for i in range(n):
        p[i+1] = p[i] + a[i]
    
    total_len = p[n]
    
    # The distance from s to t clockwise:
    # If s < t: dist = P[t] - P[s]
    # If s > t: dist = (Total_Len - P[s]) + P[t]
    # We want dist % M == 0.
    
    # Case 1: s < t
    # P[t] % M == P[s] % M
    # Case 2: s > t
    # (Total_Len + P[t] - P[s]) % M == 0  =>  P[s] % M == (Total_Len + P[t]) % M
    
    # Let's count occurrences of each remainder P[i] % M for i in 1...N
    # Note: P[1] is the distance to area 1, P[2] to area 2, etc.
    # The distance from s to t (s != t) is what matters.
    # Let X_i = P[i] % M for i = 1 to N.
    # s < t: X_t - X_s = 0 (mod M)  => X_t = X_s
    # s > t: X_t - X_s + Total_Len = 0 (mod M) => X_s = (X_t + Total_Len) (mod M)
    
    counts = [0] * m
    for i in range(1, n + 1):
        counts[p[i-1] % m] += 1
    # Wait, the indexing above is slightly off. 
    # Let's redefine: 
    # Dist(1, 1) = 0
    # Dist(1, 2) = A_1
    # Dist(1, 3) = A_1 + A_2
    # ...
    # Dist(1, N) = A_1 + ... + A_{N-1}
    # These are our P values for i=1 to N.
    
    p_mod = []
    current_sum = 0
    for i in range(n):
        p_mod.append(current_sum % m)
        current_sum += a[i]
    
    total_mod = current_sum % m
    
    # Frequency map of P[i] % M
    freq = [0] * m
    for val in p_mod:
        freq[val] += 1
        
    ans = 0
    
    # For each possible remainder r:
    # Pairs (s, t) where s < t and P[s] % M == P[t] % M:
    # This is simply combinations of freq[r] choose 2.
    for r in range(m):
        ans += (freq[r] * (freq[r] - 1)) // 2
        
    # Pairs (s, t) where s > t and P[s] % M == (P[t] % M + total_mod) % M:
    # Let r_t = P[t] % M. We need P[s] % M = (r_t + total_mod) % M.
    # Let r_s = (r_t + total_mod) % M.
    # For a fixed t, any s > t with P[s] % M = r_s works.
    # However, it's easier to think: for every pair (s, t) with s != t:
    # If s < t, condition is P[s] % M == P[t] % M.
    # If s > t, condition is P[s] % M == (P[t] % M + total_mod) % M.
    
    # Let's iterate through all t from 1 to N:
    # We need to count s such that:
    # 1. s > t and P[s] % M == P[t] % M
    # 2. s < t and P[s] % M == (P[t] % M + total_mod) % M
    
    # Let's use the frequency array for the second part.
    # For a fixed t, the number of s > t with P[s] % M == P[t] % M is handled by (freq[r] choose 2).
    # Now we need to count pairs (s, t) where s > t and P[s] % M == (P[t] % M + total_mod) % M.
    # This is equivalent to: for each t, count s in {t+1, ..., N} such that P[s] % M = (P[t] % M + total_mod) % M.
    
    # Let's re-evaluate:
    # Total pairs = Sum_{t=1 to N} [count s > t s.t. P[s] % M == P[t] % M] 
    #              + Sum_{t=1 to N} [count s > t s.t. P[s] % M == (P[t] % M + total_mod) % M]
    # This is not quite right. The problem asks for pairs (s, t) with s != t.
    # The distance is clockwise from s to t.
    # If s < t, dist = P[t] - P[s]
    # If s > t, dist = (Total - P[s]) + P[t]
    
    # Let X_i = P[i] % M.
    # Condition: 
    # s < t: X_t - X_s = 0 (mod M)  => X_t = X_s
    # s > t: X_t - X_s + Total = 0 (mod M) => X_s = (X_t + Total) (mod M)
    
    # Correct approach:
    # 1. Count pairs (s, t) with s < t and X_s == X_t.
    #    This is sum(freq[r] * (freq[r]-1) // 2)
    # 2. Count pairs (s, t) with s > t and X_s == (X_t + total_mod) % M.
    #    Iterate t from 1 to N, and maintain a count of X_s seen so far? 
    #    No, s > t means s comes after t.
    #    Iterate t from N down to 1, and maintain counts of X_s for s > t.
    
    # Let's use the freq array for s < t and a sliding count for s > t.
    # But wait, the total_mod might be 0.
    # If total_mod == 0, then X_s = X_t is the condition for both s < t and s > t.
    # That would mean any s != t with X_s == X_t works.
    # Total = freq[r] * (freq[r] - 1) for each r.
    
    if total_mod == 0:
        final_ans = 0
        for r in range(m):
            final_ans += freq[r] * (freq[r] - 1)
        print(final_ans)
        return

    # If total_mod != 0:
    # Part 1: s < t and X_s == X_t
    ans_s_lt_t = 0
    for r in range(m):
        ans_s_lt_t += (freq[r] * (freq[r] - 1)) // 2
        
    # Part 2: s > t and X_s == (X_t + total_mod) % M
    # Iterate t from N down to 1.
    ans_s_gt_t = 0
    current_freq = [0] * m
    for i in range(n - 1, -1, -1):
        xt = p_mod[i]
        xs_target = (xt + total_mod) % m
        ans_s_gt_t += current_freq[xs_target]
        current_freq[xt] += 1
        
    print(ans_s_lt_t + ans_s_gt_t)

if __name__ == '__main__':
    solve()