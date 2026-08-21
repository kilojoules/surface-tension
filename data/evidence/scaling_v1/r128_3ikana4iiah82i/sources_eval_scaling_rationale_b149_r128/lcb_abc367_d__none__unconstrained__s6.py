import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances clockwise from rest area 1
    # P[i] is the distance from area 1 to area i+1
    # P[0] = 0, P[1] = A_1, P[2] = A_1 + A_2, ...
    # We use accumulate to avoid loops
    prefixes = list(accumulate(a, lambda x, y: x + y, initial=0))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    # In both cases, we want (Dist) % M == 0
    # This is equivalent to P[t-1] % M == P[s-1] % M
    # Note: Total_Sum is prefixes[-1]
    
    # Calculate remainders modulo M for all prefix sums
    # We only need the first N prefixes (0 to N-1) because P[N] is the total sum
    # and the distance from s to t is measured clockwise.
    # Let R[i] = P[i] % M.
    # For s < t: (R[t-1] - R[s-1]) % M == 0  => R[t-1] == R[s-1]
    # For s > t: (Total_Sum - R[s-1] + R[t-1]) % M == 0 => R[s-1] - R[t-1] == Total_Sum % M
    
    # Let's simplify:
    # Let S = Total_Sum % M.
    # We seek pairs (s, t) with 1 <= s, t <= N and s != t such that:
    # If s < t: R[t-1] == R[s-1]
    # If s > t: R[s-1] - R[t-1] == S (mod M)
    
    # Let's use a frequency map of the remainders R[0]...R[N-1]
    r_values = [p % m for p in prefixes[:-1]]
    counts = Counter(r_values)
    
    # For a fixed remainder v, there are counts[v] indices.
    # The number of pairs (s, t) with s < t and R[s-1] == R[t-1] == v is:
    # counts[v] * (counts[v] - 1) // 2
    # The number of pairs (s, t) with s > t and R[s-1] - R[t-1] == S (mod M) is:
    # For each v, we need R[t-1] = (v - S) % M.
    # The number of such pairs is counts[v] * counts[(v - S) % M]
    # HOWEVER, we must exclude the case where s == t (though the problem says s != t).
    # If S == 0, then (v - S) % M == v, and the s > t case also counts pairs with the same remainder.
    
    s_mod = prefixes[-1] % m
    
    # Calculate pairs for s < t: sum of (c * (c-1) // 2)
    # Calculate pairs for s > t: sum of (counts[v] * counts[(v - s_mod) % m])
    # Note: if s_mod == 0, the s > t case is identical to the s < t case in terms of remainders.
    
    # Using map and sum to avoid loops
    ans_s_lt_t = sum(map(lambda c: c * (c - 1) // 2, counts.values()))
    
    # For s > t, we iterate over the unique remainders present in the counts
    # We need to find pairs (v, (v - s_mod) % m)
    # To avoid double counting or missing, we can just sum counts[v] * counts[(v - s_mod) % m]
    # across all v, but we must handle the case where v == (v - s_mod) % m (which happens if s_mod == 0)
    # If s_mod == 0, the condition R[s-1] - R[t-1] == 0 is the same as R[s-1] == R[t-1].
    # Since we need s > t, for each group of size c, there are c*(c-1)//2 pairs.
    
    if s_mod == 0:
        # Every pair that satisfied s < t also satisfies s > t
        print(ans_s_lt_t * 2)
    else:
        # For s > t, we need R[s-1] - R[t-1] == s_mod (mod M)
        # This means R[t-1] == (R[s-1] - s_mod) % M
        # For each v in counts, we have counts[v] choices for s and counts[(v - s_mod) % m] choices for t.
        ans_s_gt_t = sum(map(lambda v: counts[v] * counts[(v - s_mod) % m], counts.keys()))
        print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()