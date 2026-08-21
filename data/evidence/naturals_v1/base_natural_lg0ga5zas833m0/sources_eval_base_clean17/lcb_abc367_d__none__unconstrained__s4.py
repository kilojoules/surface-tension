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
    # The total distance around the lake is S = A[0] + ... + A[N-1]
    
    # Distance clockwise from s to t (s < t): P[t] - P[s]
    # Distance clockwise from s to t (s > t): S - (P[s] - P[t])
    
    # We want (Distance) % M == 0.
    # Let X[i] = P[i] % M.
    # Case 1: s < t. We need (X[t] - X[s]) % M == 0, which means X[t] == X[s].
    # Case 2: s > t. We need (S - (X[s] - X[t])) % M == 0, which means (X[s] - X[t]) % M == S % M.
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    S_mod = P[N] % M
    
    # Count occurrences of each remainder modulo M for P[1]...P[N]
    # Note: P[1] is 0, P[2] is A[0], ..., P[N] is A[0]+...+A[N-2]
    # The distance from rest area i to i+1 is A[i-1].
    # The distance from 1 to i is P[i-1].
    
    # Let's redefine: distance from rest area 1 to rest area i is dist[i]
    # dist[1] = 0
    # dist[2] = A[0]
    # ...
    # dist[N] = A[0] + ... + A[N-2]
    
    dist_mods = []
    current_sum = 0
    for i in range(N - 1):
        dist_mods.append(current_sum % M)
        current_sum += A[i]
    # This gives dists for 1 to N-1. We need to include the Nth one.
    # Actually, it's easier to just iterate:
    
    mods = [0] * N
    s = 0
    for i in range(N):
        mods[i] = s % M
        s += A[i]
    
    # total_sum_mod = s % M
    total_sum_mod = s % M
    
    # Frequency map of mods
    freq = {}
    for x in mods:
        freq[x] = freq.get(x, 0) + 1
        
    ans = 0
    
    # For each s, we look for t such that:
    # If s < t: dist[t] - dist[s] = 0 (mod M) => dist[t] = dist[s] (mod M)
    # If s > t: total_sum - (dist[s] - dist[t]) = 0 (mod M) => dist[t] = dist[s] - total_sum (mod M)
    
    # Instead of iterating s and t, we use the freq map.
    # For a fixed value v = dist[s] % M:
    # Pairs (s, t) with s < t and dist[t] % M == v:
    # If there are 'count' occurrences of v, there are count * (count - 1) // 2 such pairs.
    # But wait, the problem asks for pairs (s, t) where s != t.
    # For each s, we want to count t != s such that clockwise dist is 0 mod M.
    
    # Let's fix s and count t.
    # t can be any index in {1...N} except s.
    # If t > s: we need mods[t-1] == mods[s-1] (mod M)
    # If t < s: we need mods[t-1] == (mods[s-1] - total_sum_mod) (mod M)
    
    # This is equivalent to:
    # For each s, count t > s such that mods[t-1] == mods[s-1]
    # PLUS count t < s such that mods[t-1] == (mods[s-1] - total_sum_mod) % M
    
    # To do this efficiently:
    # 1. Count all occurrences of each mod in the whole array.
    # 2. Iterate through the array, maintaining counts of mods seen so far (left side).
    
    left_freq = {}
    total_freq = freq
    
    for i in range(N):
        v = mods[i]
        
        # t > s: t is to the right. 
        # Number of t's is (total_freq[v] - 1) - (left_freq.get(v, 0))
        # This counts t > i where mods[t] == v.
        right_count = (total_freq.get(v, 0) - 1) - left_freq.get(v, 0)
        ans += right_count
        
        # t < s: t is to the left.
        # We need mods[t] == (v - total_sum_mod) % M
        target = (v - total_sum_mod) % M
        left_count = left_freq.get(target, 0)
        ans += left_count
        
        left_freq[v] = left_freq.get(v, 0) + 1
        
    print(ans)

if __name__ == "__main__":
    solve()