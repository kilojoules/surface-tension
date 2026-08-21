import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Let P[i] be the distance from rest area 1 to rest area i+1 clockwise.
    # P[0] = 0
    # P[1] = A[0]
    # P[2] = A[0] + A[1]
    # ...
    # P[N] = A[0] + ... + A[N-1]
    
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] + A[i]
    
    total_dist = P[N]
    
    # The clockwise distance from s to t:
    # If s < t: dist = P[t-1] - P[s-1]
    # If s > t: dist = (P[N] - P[s-1]) + P[t-1]
    # We want dist % M == 0.
    
    # Let X_i = P[i] % M for i = 0 to N-1.
    # These represent the positions of rest areas 1 to N relative to area 1.
    # Note: P[0] corresponds to area 1, P[1] to area 2, ..., P[N-1] to area N.
    
    X = [P[i] % M for i in range(N)]
    
    # Count occurrences of each remainder modulo M
    count = [0] * M
    for x in X:
        count[x] += 1
    
    ans = 0
    
    # Case 1: s < t
    # dist = P[t-1] - P[s-1]
    # dist % M == 0  =>  X[t-1] == X[s-1]
    # For each remainder r, if there are count[r] positions, 
    # there are count[r] * (count[r] - 1) // 2 pairs (s, t) with s < t.
    for r in range(M):
        ans += count[r] * (count[r] - 1) // 2
        
    # Case 2: s > t
    # dist = total_dist - P[s-1] + P[t-1]
    # dist % M == 0  =>  P[t-1] - P[s-1] == -total_dist (mod M)
    # Let T = total_dist % M.
    # X[t-1] - X[s-1] == -T (mod M)  =>  X[s-1] == X[t-1] + T (mod M)
    
    T = total_dist % M
    
    # For a fixed t, we need s such that X[s-1] = (X[t-1] + T) % M and s > t.
    # However, it's easier to iterate over all pairs (s, t) where s > t.
    # Let's re-evaluate: we need pairs (s, t) such that 1 <= t < s <= N 
    # and (P[N] - P[s-1] + P[t-1]) % M == 0.
    # This is equivalent to: X[t-1] - X[s-1] == -T (mod M)
    # X[s-1] == (X[t-1] + T) % M.
    
    # To avoid double counting and handle s > t, we can iterate through X:
    # For each X[i], it can be a 't' (the destination) and we look for 's' (the start)
    # that appeared before it in the sequence (since s > t is the condition).
    # Wait, the indices are: s is start, t is end.
    # s > t means s is "further" clockwise from 1 than t is.
    # Let's use the counts.
    # For each possible value r = X[t-1], we need X[s-1] = (r + T) % M.
    # The number of pairs (s, t) with s > t is:
    # Sum over all i from 0 to N-1: (count of (X[i] + T)%M among indices j > i)
    
    # Alternatively, for every pair (s, t) with s != t:
    # If s < t, condition is X[t-1] == X[s-1]
    # If s > t, condition is X[s-1] == (X[t-1] + T) % M
    
    # Let's use the frequency array.
    # For each r1, r2 such that r1 == (r2 + T) % M:
    # we have pairs (s, t) where X[s-1] = r1 and X[t-1] = r2.
    # If r1 != r2, all such pairs have s != t.
    # If r1 == r2 (which happens if T == 0), we must ensure s != t.
    
    # Let's calculate the s > t case separately:
    # For each i from 0 to N-1 (representing t):
    # we need s > i such that X[s] = (X[i] + T) % M.
    # This is tricky because the loop is clockwise.
    
    # Let's use the property:
    # Total pairs = (Pairs s < t where X[t-1] - X[s-1] == 0 mod M)
    #             + (Pairs s > t where X[t-1] - X[s-1] == -T mod M)
    
    # Let's iterate through the array X and maintain counts of elements seen so far.
    # For Case 1 (s < t):
    # As we iterate i from 0 to N-1, X[i] is X[t-1]. We need X[s-1] == X[i] for s-1 < i.
    # For Case 2 (s > t):
    # As we iterate i from 0 to N-1, X[i] is X[s-1]. We need X[t-1] == (X[i] - T) % M for t-1 < i.
    
    ans = 0
    seen = [0] * M
    for x in X:
        # Case 1: x is X[t-1], we need X[s-1] == x
        ans += seen[x]
        # Case 2: x is X[s-1], we need X[t-1] == (x - T) % M
        ans += seen[(x - T) % M]
        
        seen[x] += 1
        
    # Special case: if T == 0, the conditions X[t-1] == X[s-1] and 
    # X[s-1] == (X[t-1] + T) % M are the same.
    # The logic above handles this: 
    # If T == 0, for each x, we add seen[x] twice.
    # But we only want to count the pair (s, t) once.
    # Wait, the problem asks for pairs (s, t). (1, 3) is different from (3, 1).
    # If T == 0, and X[s-1] == X[t-1], then both clockwise(s, t) and clockwise(t, s) 
    # are multiples of M.
    # Example: N=2, M=10, A=[5, 5]. P=[0, 5, 10]. X=[0, 5]. T=0.
    # s=1, t=2: dist=5 (no). s=2, t=1: dist=5 (no).
    # Example: N=2, M=10, A=[10, 10]. P=[0, 10, 20]. X=[0, 0]. T=0.
    # s=1, t=2: dist=10 (yes). s=2, t=1: dist=10 (yes).
    # My code: i=0: seen[0]=0, ans=0, seen[0]=1. i=1: x=0, ans += seen[0] (1) + seen[0] (1) = 2.
    # Correct.
    
    print(ans)

if __name__ == "__main__":
    solve()