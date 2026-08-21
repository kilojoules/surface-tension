import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    data_iter = iter(input_data)
    try:
        n = next(data_iter)
        m = next(data_iter)
        a = list(data_iter)
    except StopIteration:
        return

    # Calculate prefix sums of A_i modulo M.
    # P[i] is the distance from rest area 1 to rest area i+1.
    # P = [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    # We use accumulate to avoid loops.
    p = list(accumulate([0] + a, lambda x, y: (x + y) % m))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M.
    # This is 0 mod M if P[t-1] == P[s-1].
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M.
    # This is 0 mod M if (P[s-1] - P[t-1]) % M == TotalSum % M.
    
    # P[0] is 0, P[1] is A1, ..., P[N] is sum(A1...AN).
    # We are interested in indices 1 to N for the rest areas.
    # Let S = P[N] (the total sum modulo M).
    # For s < t: we need P[t-1] == P[s-1] (where indices are 1-based)
    # For s > t: we need (P[s-1] - P[t-1]) % M == S
    
    # Let's use the prefix sums P[0]...P[N-1] corresponding to rest areas 1...N.
    # The distance from area i to area j (i < j) is (P[j-1] - P[i-1]) % M.
    # The distance from area i to area j (i > j) is (P[N-1] - P[i-1] + P[j-1]) % M 
    # Wait, the definition of P above is P[0]=0, P[1]=A1... P[N]=sum(A).
    # Rest area i is at distance P[i-1] from area 1.
    # Distance i -> j (i < j): (P[j-1] - P[i-1]) % M
    # Distance i -> j (i > j): (P[N] - P[i-1] + P[j-1]) % M
    
    # Let's redefine: Prefixes of A are Pref[0]=0, Pref[1]=A1, ..., Pref[N]=sum(A1..AN)
    # Rest area i is at position Pref[i-1].
    # Pair (s, t) with s < t: (Pref[t-1] - Pref[s-1]) % M == 0  => Pref[t-1] % M == Pref[s-1] % M
    # Pair (s, t) with s > t: (Pref[N] - Pref[s-1] + Pref[t-1]) % M == 0 => (Pref[s-1] - Pref[t-1]) % M == Pref[N] % M
    
    # We only care about Pref[0]...Pref[N-1]
    prefs = p[:-1] 
    total_sum_mod = p[-1]
    
    # Count occurrences of each remainder
    counts = Counter(prefs)
    
    # For s < t: Number of pairs is sum(c * (c - 1) // 2) for each remainder c
    # For s > t: For each remainder r, we need (r - r_other) % M == total_sum_mod
    # This means r_other = (r - total_sum_mod) % M
    # The number of pairs is sum(counts[r] * counts[(r - total_sum_mod) % M])
    # BUT we must exclude the case where s == t (which is already handled by the logic, 
    # but if total_sum_mod == 0, then r == r_other, and we'd count s=t).
    
    # Using map/sum instead of loops:
    # Part 1: s < t
    ans_lt = sum(map(lambda c: c * (c - 1) // 2, counts.values()))
    
    # Part 2: s > t
    # We need to calculate sum(counts[r] * counts[(r - total_sum_mod) % m])
    # and subtract cases where s == t (which happens if total_sum_mod == 0)
    ans_gt = sum(map(lambda r: counts[r] * counts[(r - total_sum_mod) % m], counts.keys()))
    
    # If total_sum_mod == 0, the s > t logic includes s == t. 
    # There are N such cases.
    if total_sum_mod == 0:
        ans_gt -= n
        
    print(ans_lt + ans_gt)

if __name__ == "__main__":
    solve()