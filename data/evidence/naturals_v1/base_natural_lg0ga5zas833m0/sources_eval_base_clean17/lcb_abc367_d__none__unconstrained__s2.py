import sys

def solve():
    # Read N and M from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, m = map(int, line1)
        
        # Read A_i from the following lines
        a = []
        while len(a) < n:
            line = sys.stdin.readline().split()
            if not line:
                break
            a.extend(map(int, line))
    except ValueError:
        return

    # Let P[i] be the prefix sum of A: distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2
    # ...
    # P[N] = A_1 + ... + A_N
    
    p = [0] * (n + 1)
    for i in range(n):
        p[i+1] = p[i] + a[i]
    
    # The distance clockwise from s to t:
    # If s < t: distance = P[t-1] - P[s-1]
    # If s > t: distance = (P[N] - P[s-1]) + P[t-1]
    
    # We want distance % M == 0.
    # Let x_i = P[i-1] % M for i = 1 to N.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  =>  P[t-1] % M == P[s-1] % M
    # Case 2: s > t
    # (P[N] - P[s-1] + P[t-1]) % M == 0  =>  P[t-1] % M == (P[s-1] - P[N]) % M
    
    # Let's define V[i] = P[i] % M for i = 0 to N-1.
    # These represent the distances from rest area 1 to rest area 1, 2, ..., N.
    # V = [P[0]%M, P[1]%M, ..., P[N-1]%M]
    v = [p[i] % m for i in range(n)]
    
    # Count occurrences of each remainder
    count = {}
    for x in v:
        count[x] = count.get(x, 0) + 1
        
    total_sum_mod = p[n] % m
    ans = 0
    
    # For each possible remainder r = P[s-1] % M:
    # We look for t such that:
    # 1. s < t and P[t-1] % M == r
    # 2. s > t and P[t-1] % M == (r - total_sum_mod) % M
    
    # To handle s < t and s > t efficiently:
    # Let's iterate through all possible remainders r that exist in the set.
    # For a fixed r, let C(r) be the number of i in {0...N-1} such that P[i]%M == r.
    # For any two indices i, j in {0...N-1} with i < j:
    # The pair (s,t) = (i+1, j+1) is valid if P[j]%M == P[i]%M.
    # This gives C(r) * (C(r) - 1) // 2 pairs for each r.
    
    # For the case s > t:
    # Let s-1 = i and t-1 = j, where i > j.
    # (P[N] - P[i] + P[j]) % M == 0  =>  P[j] % M == (P[i] - P[N]) % M.
    # For a fixed i, we need to count j < i such that P[j] % M == (P[i] - total_sum_mod) % M.
    
    # Actually, a simpler way:
    # For every pair (i, j) with 0 <= i < j < N:
    # Pair (i+1, j+1) is valid if (P[j] - P[i]) % M == 0.
    # Pair (j+1, i+1) is valid if (P[N] - P[j] + P[i]) % M == 0.
    
    # Let's use the counts:
    # For a fixed remainder r1 and r2:
    # Number of pairs (i, j) with i < j such that P[i]%M = r1 and P[j]%M = r2 is:
    # This is tricky because of the i < j constraint.
    
    # Let's reconsider:
    # Total pairs = Sum_{i < j} [ (P[j]-P[i])%M == 0 ] + Sum_{i < j} [ (P[N]-P[j]+P[i])%M == 0 ]
    # First term: For each r, C(r) * (C(r)-1) // 2
    # Second term: Sum_{i < j} [ P[i]%M == (P[j] - P[N])%M ]
    
    # Let's calculate the second term by iterating through the array:
    # For j = 0 to N-1:
    #   target = (P[j] - total_sum_mod) % M
    #   ans += count_of_target_seen_so_far
    #   update count_of_P[j]%M_seen_so_far
    
    # First term
    for r in count:
        c = count[r]
        ans += c * (c - 1) // 2
        
    # Second term
    seen = {}
    for j in range(n):
        target = (v[j] - total_sum_mod) % m
        ans += seen.get(target, 0)
        seen[v[j]] = seen.get(v[j], 0) + 1
        
    print(ans)

if __name__ == "__main__":
    solve()