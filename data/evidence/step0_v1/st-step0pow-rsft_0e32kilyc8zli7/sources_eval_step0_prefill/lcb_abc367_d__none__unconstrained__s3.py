import sys

def solve():
    # Read N and M from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the prefix sum of A, where P[0] = 0.
    # P[i] = A_1 + ... + A_i.
    # The distance from rest area s to t (s < t) is P[t-1] - P[s-1].
    # The distance from rest area s to t (s > t) is (P[N] - P[s-1]) + P[t-1].
    # We want distance % M == 0.
    
    # Let X_i = P[i] % M.
    # For s < t: (X_{t-1} - X_{s-1}) % M == 0  => X_{t-1} == X_{s-1}
    # For s > t: (X_N - X_{s-1} + X_{t-1}) % M == 0 => (X_{s-1} - X_{t-1}) % M == X_N % M
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    X = [P[i] % M for i in range(N)]
    # X[i] corresponds to the distance from area 1 to area i+1.
    # Note: area 1 is index 0, area N is index N-1.
    
    # Count occurrences of each remainder
    count = {}
    for val in X:
        count[val] = count.get(val, 0) + 1
        
    total_pairs = 0
    X_N = P[N] % M
    
    # Case 1: s < t
    # We need X[t-1] == X[s-1]. 
    # For each remainder r, if it appears c times, there are c*(c-1)//2 pairs.
    for r in count:
        c = count[r]
        total_pairs += c * (c - 1) // 2
        
    # Case 2: s > t
    # We need (X[s-1] - X[t-1]) % M == X_N % M
    # Which is X[s-1] % M == (X[t-1] + X_N) % M
    # We iterate over all possible remainders r for X[t-1].
    for r in count:
        c_t = count[r]
        r_s = (r + X_N) % M
        if r_s in count:
            c_s = count[r_s]
            # We need to count pairs (s, t) where s > t.
            # This is tricky because the simple product c_s * c_t includes cases where s < t.
            # Let's rethink.
            pass

    # Correct approach for s > t:
    # We want to count pairs (i, j) such that 0 <= j < i < N and (X[i] - X[j]) % M == X_N % M.
    # This is equivalent to X[i] % M == (X[j] + X_N) % M.
    # We can iterate through the array X and keep track of counts of remainders seen so far.
    
    # Reset total_pairs and calculate both cases in one pass or separately.
    # Let's use the property:
    # Total pairs = (pairs s < t where X[t-1] == X[s-1]) + (pairs s > t where X[s-1] - X[t-1] == X_N mod M)
    
    # For s < t:
    ans = 0
    # For s > t:
    # We need X[s-1] - X[t-1] \equiv X_N (mod M)
    # Let's use a frequency map for X values.
    # For a fixed s, we need X[t-1] \equiv X[s-1] - X_N (mod M) where t < s.
    
    # To handle both s < t and s > t:
    # s < t: X[t-1] == X[s-1]
    # s > t: X[t-1] == (X[s-1] - X_N) % M
    
    # Let's just use the frequency map for s < t and a loop for s > t.
    ans_s_lt_t = 0
    for r in count:
        c = count[r]
        ans_s_lt_t += c * (c - 1) // 2
        
    ans_s_gt_t = 0
    prefix_counts = {}
    for i in range(N):
        # i is the index for s-1. We look for t-1 < i.
        target = (X[i] - X_N) % M
        ans_s_gt_t += prefix_counts.get(target, 0)
        prefix_counts[X[i]] = prefix_counts.get(X[i], 0) + 1
        
    print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()