import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances from rest area 1.
    # P[i] = distance from rest area 1 to rest area (i+1)
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    # The distance from s to t clockwise is:
    # If s < t: P[t-1] - P[s-1]
    # If s > t: (P[N] - P[s-1]) + P[t-1]
    # We want this distance to be 0 mod M.
    
    # Let X_i = P[i-1] mod M.
    # For s < t: (X_t - X_s) % M == 0  =>  X_t == X_s
    # For s > t: (P[N] - X_s + X_t) % M == 0 => X_s - X_t == P[N] % M
    
    # Count occurrences of each remainder mod M
    count = [0] * M
    for i in range(1, N + 1):
        count[P[i-1] % M] += 1
        
    total_pairs = 0
    S_total = P[N] % M
    
    # Case 1: s < t
    # For a fixed remainder r, if there are c elements with that remainder,
    # there are c * (c-1) / 2 pairs where s < t and X_s == X_t.
    # However, we can just iterate through the counts.
    for r in range(M):
        c = count[r]
        total_pairs += c * (c - 1) // 2
        
    # Case 2: s > t
    # We need X_s - X_t = S_total (mod M), which is X_t = (X_s - S_total) (mod M)
    # For each s, we look for t < s. But it's easier to think globally:
    # We want the number of pairs (s, t) such that s > t and X_s - X_t = S_total (mod M).
    # This is tricky because the condition s > t depends on the order.
    # Let's reconsider:
    # Total pairs (s, t) with s != t is N * (N-1).
    # Let's use the property:
    # Distance(s, t) = (P[t-1] - P[s-1]) mod P[N] (conceptually)
    # Actually, distance(s, t) = (P[t-1] - P[s-1]) mod (Sum of all A)
    # But the problem says "minimum number of steps to walk clockwise".
    # Clockwise distance from s to t:
    # if s < t: Dist = P[t-1] - P[s-1]
    # if s > t: Dist = P[N] - P[s-1] + P[t-1]
    
    # Let X_i = P[i-1] % M.
    # Condition:
    # if s < t: X_t - X_s \equiv 0 (mod M)  => X_t \equiv X_s (mod M)
    # if s > t: X_t - X_s + S_total \equiv 0 (mod M) => X_s - X_t \equiv S_total (mod M)
    
    # Let's re-evaluate the total count using the frequency array:
    # For a fixed remainder r, there are count[r] indices.
    # Any two indices i, j with the same remainder r:
    # If i < j, then s=i, t=j satisfies the condition.
    # If i > j, then s=i, t=j satisfies the condition IF S_total == 0.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t:
    # If s < t, we need X_s == X_t.
    # If s > t, we need X_s - X_t == S_total (mod M).
    
    # Let's iterate over all possible remainders r for X_t:
    # For a fixed t, we want s < t such that X_s = X_t
    # AND s > t such that X_s = (X_t + S_total) % M.
    
    # Total = \sum_{t=1}^N [ (count of s < t with X_s = X_t) + (count of s > t with X_s = (X_t + S_total)%M) ]
    # Total = \sum_{r=0}^{M-1} [ (count[r] * (count[r]-1) / 2) + (count[(r + S_total)%M] * count[r] - (overlap if s > t and s < t)) ]
    # This is getting confusing. Let's simplify:
    
    # Correct Logic:
    # We want pairs (s, t) with s != t such that:
    # 1. s < t and X_s == X_t
    # 2. s > t and X_s - X_t == S_total (mod M)
    
    # Let's count how many s < t satisfy X_s == X_t:
    # For each r, if count[r] = c, there are c*(c-1)//2 such pairs.
    
    # Now count how many s > t satisfy X_s - X_t == S_total (mod M):
    # This is equivalent to X_s - S_total == X_t (mod M).
    # For a fixed s, we need t < s such that X_t == (X_s - S_total) % M.
    # This can be solved by iterating t from 1 to N and maintaining counts.
    
    ans = 0
    # Part 1: s < t and X_s == X_t
    for r in range(M):
        ans += count[r] * (count[r] - 1) // 2
        
    # Part 2: s > t and X_s - X_t == S_total (mod M)
    # We need to count pairs (s, t) such that 1 <= t < s <= N and X_t == (X_s - S_total) % M
    current_counts = [0] * M
    for i in range(1, N + 1):
        x_s = P[i-1] % M
        target_x_t = (x_s - S_total) % M
        ans += current_counts[target_x_t]
        current_counts[x_s] += 1
        
    print(ans)

if __name__ == '__main__':
    solve()