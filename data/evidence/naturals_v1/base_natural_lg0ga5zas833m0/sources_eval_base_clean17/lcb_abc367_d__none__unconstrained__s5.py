import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    m = int(input_data[1])
    a = list(map(int, input_data[2:]))
    
    # Let P[i] be the prefix sum of steps from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2
    # ...
    # P[n] = A_1 + ... + A_n
    # The distance clockwise from s to t is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (P[n] - P[s-1]) + P[t-1]
    
    p = [0] * (n + 1)
    for i in range(n):
        p[i+1] = p[i] + a[i]
    
    # Total length of the lake circumference
    total_len = p[n]
    
    # We want (distance from s to t) % M == 0.
    # Let x_i = P[i-1] % M for i = 1 to N.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  =>  P[t-1] % M == P[s-1] % M
    # Case 2: s > t
    # (total_len - P[s-1] + P[t-1]) % M == 0  =>  P[t-1] % M == (P[s-1] - total_len) % M
    
    # Let's simplify. We are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Let v_i = P[i-1] % M.
    # Distance s -> t is:
    # If s < t: (v_t - v_s) % M == 0  => v_t == v_s
    # If s > t: (total_len - v_s + v_t) % M == 0 => v_t == (v_s - total_len) % M
    
    # Let's use a frequency map for v_i.
    counts = {}
    for i in range(n):
        val = p[i] % m
        counts[val] = counts.get(val, 0) + 1
        
    ans = 0
    # For each s, we want to find how many t satisfy the condition.
    # Let v_s = P[s-1] % M.
    # For t > s: we need v_t == v_s.
    # For t < s: we need v_t == (v_s - total_len) % M.
    
    # This is equivalent to:
    # For every pair (s, t) with s < t: if v_s == v_t, it's a valid pair.
    # For every pair (s, t) with s > t: if v_t == (v_s - total_len) % M, it's a valid pair.
    
    # Part 1: s < t
    # For each unique value v, if it appears k times, there are k*(k-1)//2 pairs.
    for val in counts:
        k = counts[val]
        ans += k * (k - 1) // 2
        
    # Part 2: s > t
    # This is symmetric to s < t but with a shift.
    # We need v_t == (v_s - total_len) % M.
    # Let shift = total_len % M.
    # We need v_t == (v_s - shift) % M.
    # For a fixed v_s, the number of t < s such that v_t == (v_s - shift) % M
    # is tricky because of the t < s constraint.
    
    # Let's reconsider:
    # Total pairs (s, t) such that s != t and dist(s, t) % M == 0.
    # dist(s, t) = (P[t-1] - P[s-1]) if s < t else (P[n] - P[s-1] + P[t-1])
    # In both cases, dist(s, t) = (P[t-1] - P[s-1]) mod P[n] (conceptually)
    # Actually: dist(s, t) % M = (P[t-1] - P[s-1]) % M if s < t
    # and dist(s, t) % M = (P[n] + P[t-1] - P[s-1]) % M if s > t.
    
    # Let v_i = P[i-1] % M.
    # If s < t: v_t - v_s \equiv 0 (mod M)  => v_t = v_s
    # If s > t: v_t - v_s + (total_len % M) \equiv 0 (mod M) => v_t = (v_s - total_len) % M
    
    # Let L = total_len % M.
    # We want to count pairs (s, t) such that:
    # 1. s < t and v_s = v_t
    # 2. s > t and v_t = (v_s - L) % M
    
    # Let's iterate through the array v and maintain counts of values seen so far.
    # For i = 1 to N:
    #   v_i = P[i-1] % M
    #   // This i acts as 't' for s < t:
    #   ans += count[v_i]
    #   // This i acts as 's' for s > t:
    #   ans += count[(v_i - L) % M]
    #   count[v_i] += 1
    
    # Wait, the logic above counts:
    # For a fixed i, it counts j < i such that v_j = v_i (Case s < t, s=j, t=i)
    # AND it counts j < i such that v_j = (v_i - L) % M (Case s > t, s=i, t=j)
    
    # Let's trace:
    # For i = 1 to N:
    #   v = P[i-1] % M
    #   ans += current_counts[v] # s < t, s is some j < i, t is i
    #   ans += current_counts[(v - L) % M] # s > t, s is i, t is some j < i
    #   current_counts[v] += 1
    
    # Special case: if L == 0, then (v - L) % M == v.
    # The two conditions become the same: v_s = v_t.
    # If L == 0, the loop adds current_counts[v] twice.
    # But the problem says s != t. 
    # If L == 0, dist(s, t) % M = (v_t - v_s) % M.
    # So we need v_t = v_s.
    # For every pair {s, t} with s != t, there are two ordered pairs (s, t) and (t, s).
    # Both will have dist % M = 0 if v_s = v_t.
    # Total = k * (k-1) for each unique value.
    
    # Let's re-run the logic with L:
    # For i = 1 to N:
    #   v = P[i-1] % M
    #   ans += current_counts[v]
    #   if L != 0:
    #       ans += current_counts[(v - L) % M]
    #   else:
    #       # If L == 0, then (v-L)%M == v.
    #       # s < t and s > t both require v_s == v_t.
    #       # The first 'ans += current_counts[v]' handles s < t.
    #       # We need to handle s > t separately.
    #       # Actually, if L == 0, dist(s, t) % M = (v_t - v_s) % M.
    #       # For any s != t, if v_s == v_t, then both (s, t) and (t, s) are valid.
    #       # So we just multiply the k*(k-1)//2 by 2.
    #       pass
    #   current_counts[v] += 1
    
    # Correct logic for L == 0:
    # if L == 0:
    #     ans = 0
    #     for val in counts:
    #         k = counts[val]
    #         ans += k * (k - 1)
    # else:
    #     # use the loop
    
    L = total_len % m
    if L == 0:
        final_ans = 0
        for val in counts:
            k = counts[val]
            final_ans += k * (k - 1)
        print(final_ans)
    else:
        final_ans = 0
        current_counts = {}
        for i in range(n):
            v = p[i] % m
            final_ans += current_counts.get(v, 0)
            final_ans += current_counts.get((v - L) % m, 0)
            current_counts[v] = current_counts.get(v, 0) + 1
        print(final_ans)

if __name__ == "__main__":
    solve()