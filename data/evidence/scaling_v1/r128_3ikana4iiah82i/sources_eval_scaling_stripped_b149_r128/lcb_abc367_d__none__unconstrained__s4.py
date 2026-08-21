import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate the prefix sums of distances modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A[0], P[2] = A[0] + A[1]...
    # We use accumulate to avoid loops and map to keep values within [0, M-1]
    P = list(accumulate(A, lambda x, y: (x + y) % M))
    # We need the distance from 1 to 1 to be 0 for the calculation
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    
    # Let S be the total sum of A modulo M
    S = sum(A) % M
    
    # We are looking for pairs (s, t) such that distance is 0 mod M.
    # For s < t: P[t-1] - P[s-1] = 0 mod M  => P[t-1] = P[s-1] mod M
    # For s > t: S - P[s-1] + P[t-1] = 0 mod M => P[s-1] - P[t-1] = S mod M
    
    # Let's collect all P values including the starting point 0
    # There are N points: index 0 to N-1.
    # The distance from point i to point j (i < j) is (P[j] - P[i]) % M
    # Note: P as defined by accumulate has N elements. 
    # Let's prepend 0 to represent the distance from area 1 to area 1.
    all_P = [0] + P[:-1] 
    # Now all_P has N elements. all_P[i] is distance from area 1 to area i+1.
    
    counts = Counter(all_P)
    
    # For s < t, we need all_P[s-1] == all_P[t-1]
    # The number of such pairs is sum(count * (count - 1) // 2)
    ans_st = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t, we need all_P[s-1] - all_P[t-1] == S mod M
    # This is equivalent to all_P[s-1] == (all_P[t-1] + S) mod M
    # We iterate through the unique values in counts to find pairs
    # Note: if S == 0, the condition is the same as s < t, but the problem 
    # says s != t. If S == 0, then for every pair (s, t) where s < t, 
    # the distance from t to s is also 0 mod M.
    
    if S == 0:
        # If total sum is 0 mod M, then for every pair (s, t) with s < t,
        # both clockwise s->t and t->s are 0 mod M.
        ans_ts = ans_st
    else:
        # For s > t, we need all_P[s-1] == (all_P[t-1] + S) % M
        # We can calculate this by summing counts[v] * counts[(v + S) % M]
        ans_ts = sum(counts[v] * counts.get((v + S) % M, 0) for v in counts)

    print(ans_st + ans_ts)

if __name__ == "__main__":
    solve()