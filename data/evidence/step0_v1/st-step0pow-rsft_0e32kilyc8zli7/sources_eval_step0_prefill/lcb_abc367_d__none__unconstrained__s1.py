import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P[i] be the prefix sum of A, where P[0] = 0.
    # P[i] = A_1 + A_2 + ... + A_i.
    # The distance from rest area s to t (s < t) is P[t-1] - P[s-1].
    # The distance from rest area s to t (s > t) is (P[N] - P[s-1]) + P[t-1].
    # We want these distances to be 0 modulo M.
    
    # Let X_i = P[i] mod M.
    # For s < t: (X_{t-1} - X_{s-1}) % M == 0  => X_{t-1} == X_{s-1}
    # For s > t: (X_N - X_{s-1} + X_{t-1}) % M == 0 => (X_{s-1} - X_{t-1}) % M == X_N % M
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    # We are interested in X_i = P[i] % M for i = 0, 1, ..., N-1.
    # Note: Rest area i corresponds to index i-1 in the 0-indexed prefix sum array.
    # Let's define V_i = P[i] % M for i = 0, ..., N-1.
    # There are N such values.
    
    V = [P[i] % M for i in range(N)]
    
    # Count occurrences of each remainder
    count = {}
    for v in V:
        count[v] = count.get(v, 0) + 1
        
    total_sum_mod = P[N] % M
    ans = 0
    
    # Case 1: s < t
    # We need V_{t-1} == V_{s-1}. 
    # For each remainder r, if it appears c times, there are c*(c-1)//2 pairs.
    for r in count:
        c = count[r]
        ans += c * (c - 1) // 2
        
    # Case 2: s > t
    # We need (V_{s-1} - V_{t-1}) % M == total_sum_mod.
    # This is equivalent to V_{s-1} == (V_{t-1} + total_sum_mod) % M.
    # We iterate over all possible V_{t-1} (which is r) and find how many V_{s-1} match.
    # However, we must ensure s > t. 
    # A simpler way: 
    # For every pair (s, t) with s != t, exactly one of (s < t) or (s > t) is true.
    # Let's just iterate over all r and find its complement.
    
    # For a fixed t, we need V_{s-1} = (V_{t-1} + total_sum_mod) % M.
    # Let r1 = V_{t-1} and r2 = (r1 + total_sum_mod) % M.
    # The number of pairs (s, t) such that s > t and the condition holds is:
    # We can use the property that for any two distinct indices i, j in {0, ..., N-1},
    # one is smaller than the other.
    # Let's use a different approach for s > t.
    # For each i in 0...N-1, we want to find j in 0...N-1 such that i > j 
    # and (V[i] - V[j]) % M == total_sum_mod.
    
    # To calculate this efficiently:
    # Iterate through the array V, keeping track of the counts of remainders seen so far.
    current_counts = {}
    for i in range(N):
        # We look for j < i such that (V[i] - V[j]) % M == total_sum_t
        # V[j] = (V[i] - total_sum_mod) % M
        target = (V[i] - total_sum_mod) % M
        ans += current_counts.get(target, 0)
        
        current_counts[V[i]] = current_counts.get(V[i], 0) + 1

    # Wait, the logic above for s < t was: V_{t-1} == V_{s-1}.
    # Let's re-evaluate.
    # Let i = s-1 and j = t-1. i, j \in {0, ..., N-1}, i != j.
    # If i < j: distance is (P[j] - P[i]) % M == 0  => V[j] == V[i]
    # If i > j: distance is (P[N] - P[i] + P[j]) % M == 0 => (V[i] - V[j]) % M == P[N] % M
    
    # Let's clear ans and redo.
    ans = 0
    # For i < j:
    # We can count this by iterating and keeping track of seen V[i].
    seen = {}
    for j in range(N):
        val = V[j]
        ans += seen.get(val, 0)
        seen[val] = seen.get(val, 0) + 1
        
    # For i > j:
    # We need (V[i] - V[j]) % M == total_sum_mod.
    # As we iterate i from 0 to N-1, we count how many j < i satisfy V[j] == (V[i] - total_sum_mod) % M.
    seen_for_gt = {}
    for i in range(N):
        target = (V[i] - total_sum_mod) % M
        ans += seen_for_gt.get(target, 0)
        seen_for_gt[V[i]] = seen_for_gt.get(V[i], 0) + 1
        
    # Special case: if total_sum_mod == 0, then the condition for i > j 
    # (V[i] - V[j]) % M == 0 is the same as V[i] == V[j].
    # But the problem says s != t, so i != j.
    # The logic above handles total_sum_mod == 0 correctly because it only counts j < i.
    
    # However, there is a catch: the "minimum number of steps to walk clockwise"
    # from s to t is simply the sum of A_k from k=s to t-1 (if s < t)
    # or from k=s to N and k=1 to t-1 (if s > t).
    # This is exactly what P[j] - P[i] and (P[N] - P[i]) + P[j] represent.
    
    # Let's double check Sample 1: N=4, M=3, A=[2, 1, 4, 3]
    # P = [0, 2, 3, 7, 10]
    # V = [0, 2, 0, 1] (P[0..3] % 3)
    # total_sum_mod = 10 % 3 = 1
    # i < j:
    # j=0: V[0]=0, seen={} -> ans=0, seen={0:1}
    # j=1: V[1]=2, seen={0:1} -> ans=0, seen={0:1, 2:1}
    # j=2: V[2]=0, seen={0:1, 2:1} -> ans=1, seen={0:2, 2:1}
    # j=3: V[3]=1, seen={0:2, 2:1} -> ans=1, seen={0:2, 2:1, 1:1}
    # i > j: (V[i] - V[j]) % 3 == 1  => V[j] == (V[i] - 1) % 3
    # i=0: V[0]=0, target=(0-1)%3=2, seen={} -> ans=1, seen={0:1}
    # i=1: V[1]=2, target=(2-1)%3=1, seen={0:1} -> ans=1, seen={0:1, 2:1}
    # i=2: V[2]=0, target=(0-1)%3=2, seen={0:1, 2:1} -> ans=2, seen={0:2, 2:1}
    # i=3: V[3]=1, target=(1-1)%3=0, seen={0:2, 2:1} -> ans=4, seen={0:2, 2:1, 1:1}
    # Result: 4. Correct.
    
    print(ans)

if __name__ == "__main__":
    solve()