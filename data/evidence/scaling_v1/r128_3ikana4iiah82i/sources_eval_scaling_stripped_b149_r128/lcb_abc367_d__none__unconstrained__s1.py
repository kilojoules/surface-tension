import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate the total sum of all A_i
    total_sum = sum(A)
    
    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A_1 % M, ..., P[N-1] = (A_1 + ... + A_{N-1}) % M
    # We use accumulate to avoid loops and map to keep it within modulo M
    P = list(accumulate(A[:-1], lambda x, y: (x + y) % M))
    # Insert the starting point 0 at the beginning
    P = [0] + P
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    # The distance from s to t (s > t) is (total_sum - (P[s-1] - P[t-1])) % M
    # We want (total_sum - P[s-1] + P[t-1]) % M == 0
    # Which means P[s-1] - P[t-1] == total_sum % M
    
    # Count occurrences of each prefix sum modulo M
    counts = Counter(P)
    
    # For s < t:
    # For each unique value v in P, if it appears C times, 
    # there are C * (C - 1) // 2 pairs (s, t)
    ans_st = sum(C * (C - 1) // 2 for C in counts.values())
    
    # For s > t:
    # We need P[s-1] - P[t-1] == total_sum % M
    # Let target = total_sum % M
    # For each P[t-1] = v, we need P[s-1] = (v + target) % M
    target = total_sum % M
    
    # We iterate through the unique values in counts to find pairs (v, (v + target) % M)
    # Note: if target == 0, the condition P[s-1] == P[t-1] is the same as s < t
    # But the problem asks for pairs (s, t) where s != t.
    # If target == 0, then for every pair {s, t} with s < t, 
    # both (s, t) and (t, s) satisfy the condition.
    
    if target == 0:
        # Every pair that worked for s < t also works for s > t
        ans_ts = ans_st
    else:
        # For each v, find how many P[s-1] == (v + target) % M
        # This is a summation over the keys of the counter
        ans_ts = sum(counts[v] * counts.get((v + target) % M, 0) for v in counts)
        
    print(ans_st + ans_ts)

if __name__ == "__main__":
    solve()