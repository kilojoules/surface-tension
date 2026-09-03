import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    # Parse A_i
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M
    # P[i] = (A_1 + ... + A_{i-1}) mod M
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M
    # Note: accumulate returns an iterator, we convert to list
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    # P now has N+1 elements. P[0] is the dummy 0, P[1...N] are the prefix sums.
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) mod M.
    # The distance from s to t (s > t) is (P[N] - P[s-1] + P[t-1]) mod M.
    
    # We want (dist) % M == 0.
    # Case 1: s < t => P[t-1] ≡ P[s-1] (mod M)
    # Case 2: s > t => P[s-1] ≡ (P[N] + P[t-1]) (mod M)
    
    # Let's use the indices 0 to N-1 to represent rest areas 1 to N.
    # Prefix sums S[i] = sum(A[0...i-1]) % M for i in 0...N.
    # S = [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    S = P[1:] # This is S[1...N]. Let's prepend 0.
    S = [0] + S[:-1] # S[i] is distance from area 1 to area i+1.
    # S = [0, A1%M, (A1+A2)%M, ..., (A1+...+A_{N-1})%M]
    
    # Total distance L = sum(A) % M
    L = P[N]
    
    # Count occurrences of each remainder in S
    counts = Counter(S)
    
    # For a fixed s, we need t such that:
    # If s < t: S[t-1] ≡ S[s-1] (mod M)
    # If s > t: S[s-1] ≡ L + S[t-1] (mod M) => S[t-1] ≡ S[s-1] - L (mod M)
    
    # Total pairs = sum_{s=1}^N (count of t > s where S[t-1]==S[s-1]) 
    #              + sum_{s=1}^N (count of t < s where S[t-1]==(S[s-1]-L)%M)
    
    # The first term is simply the number of ways to choose 2 indices with same S value:
    # sum(c * (c - 1) // 2 for c in counts.values())
    
    # The second term:
    # For each s, we need t < s such that S[t-1] = (S[s-1] - L) % M.
    # We can iterate through S and maintain a running count of remainders seen so far.
    
    # To avoid loops, we can use a mathematical approach for the second term:
    # Let C(v) be the count of value v in S.
    # The second term is sum_{v} (C(v) * C((v - L) % M))
    # BUT, this counts pairs (s, t) where s > t. 
    # If L == 0, then (v - L) % M == v, and we get C(v)^2.
    # However, we must ensure s != t.
    # If L == 0, the condition s < t and s > t both become S[s-1] == S[t-1].
    # Total pairs = C(v) * (C(v) - 1) for each v.
    
    # If L != 0:
    # Pairs (s, t) with s < t: S[s-1] == S[t-1] -> C(v) * (C(v)-1) // 2
    # Pairs (s, t) with s > t: S[t-1] == (S[s-1] - L) % M -> C(v) * C((v-L)%M)
    # Note: since L != 0, v != (v-L)%M, so s and t are automatically distinct.
    
    term1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    if L == 0:
        # If L is 0, then (S[s-1] - L)%M is just S[s-1].
        # The condition s > t also becomes S[s-1] == S[t-1].
        # Total = 2 * term1
        print(2 * term1)
    else:
        # term2 = sum(C(v) * C((v-L)%M))
        # We can iterate over the unique keys of the counter
        term2 = sum(counts[v] * counts.get((v - L) % M, 0) for v in counts)
        print(term1 + term2)

if __name__ == "__main__":
    solve()