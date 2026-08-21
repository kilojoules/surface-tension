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
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate(a, lambda x, y: (x + y) % m, initial=0) gives P[0]...P[N]
    # Note: initial is available in Python 3.8+
    p = list(accumulate(a, lambda x, y: (x + y) % m, initial=0))
    
    # The total sum of A_i modulo M
    total_sum_mod = p[n]
    
    # We are looking for pairs (s, t) such that dist(s, t) % M == 0.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # Case 2: s > t
    # (Total_Sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == Total_Sum % M
    
    # Count frequencies of each remainder in P[0]...P[N-1]
    # P[N] is the total sum, we only care about the starting positions 1 to N.
    counts = Counter(p[:n])
    
    # For Case 1 (s < t):
    # For each remainder r, if there are c copies, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If P[s-1] == P[t-1], then dist(s, t) is a multiple of M.
    # This covers all s < t.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For Case 2 (s > t):
    # We need (P[s-1] - P[t-1]) % M == total_sum_mod
    # Which is P[s-1] % M == (P[t-1] + total_sum_mod) % M
    # We iterate over all possible remainders r = P[t-1] % M
    # and multiply count(r) by count((r + total_sum_mod) % M).
    # Special care: if total_sum_mod == 0, this is the same as Case 1.
    # But the condition s > t must be strictly maintained.
    
    # To handle s > t, we can use the property:
    # Total pairs (s, t) such that dist(s, t) % M == 0 is:
    # Sum_{r=0 to M-1} (count(r) * count((r + total_sum_mod) % M))
    # BUT we must exclude cases where s == t (not allowed)
    # and handle the wrap-around logic.
    
    # Let's use a more direct approach for s > t:
    # For a fixed t, we need P[s-1] % M == (P[t-1] + total_sum_mod) % M
    # The number of such s is counts[(P[t-1] + total_sum_mod) % M].
    # This includes cases where s < t, s == t, and s > t.
    # Actually, the simplest way:
    # For every pair (s, t) with s != t:
    # If s < t, condition is P[t-1] - P[s-1] \equiv 0 mod M
    # If s > t, condition is P[t-1] - P[s-1] \equiv -TotalSum mod M
    
    # Let's calculate:
    # For each r, let c1 = counts[r] and c2 = counts[(r + total_sum_mod) % m]
    # If total_sum_mod == 0:
    #   Every pair (s, t) with P[s-1] == P[t-1] works.
    #   There are c*(c-1) such pairs for each r.
    # If total_sum_mod != 0:
    #   s < t works if P[s-1] == P[t-1] -> c*(c-1)//2 pairs
    #   s > t works if P[s-1] == (P[t-1] + total_sum_mod) % m -> c1 * c2 pairs
    
    # Wait, the logic for s > t is:
    # Dist(s, t) = (Total - P[s-1]) + P[t-1]
    # Dist(s, t) \equiv 0 mod M  => P[s-1] - P[t-1] \equiv Total mod M
    # => P[s-1] \equiv (P[t-1] + Total) mod M
    
    # Let's use a generator to sum the valid pairs:
    # For each r in counts:
    #   Ways to pick s < t: if P[s-1] == P[t-1] == r, then (c*(c-1)//2)
    #   Ways to pick s > t: if P[t-1] == r and P[s-1] == (r + total_sum_mod) % m, then c(r) * c((r+total_sum_mod)%m)
    #   BUT if total_sum_mod == 0, the s > t condition is also P[s-1] == P[t-1], 
    #   which is c*(c-1)//2.
    
    # Correct logic:
    # Total = Sum_{r} [ count(r)*count(r)//2 if s < t ] + Sum_{r} [ count(r)*count((r+total_sum_mod)%m) if s > t ]
    # If total_sum_mod == 0, the second term is Sum count(r)*count(r). 
    # But we must exclude s=t, and the s > t requirement.
    # Actually, if total_sum_mod == 0, then Dist(s, t) % M == 0 iff P[s-1] == P[t-1].
    # For each group of size c, there are c*(c-1) pairs.
    
    # If total_sum_mod != 0:
    # s < t: P[s-1] == P[t-1]  => sum(c*(c-1)//2)
    # s > t: P[s-1] == (P[t-1] + total_sum_mod) % m => sum(count(r) * count((r + total_sum_mod) % m))
    
    # Let's implement this using a conditional expression:
    result = (
        sum(c * (c - 1) for c in counts.values()) 
        if total_sum_mod == 0 
        else sum(c * (c - 1) // 2 for c in counts.values()) + 
             sum(counts[r] * counts.get((r + total_sum_mod) % m, 0) for r in counts)
    )
    
    print(result)

if __name__ == "__main__":
    solve()