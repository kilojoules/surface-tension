import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension to capture the rest.
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (clockwise) is:
    # If s < t: (P_t - P_s)
    # If s > t: (Total_Sum - P_s) + P_t
    
    # Calculate prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + a[:-1]) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums.
    prefixes = list(accumulate([0] + a[:-1]))
    total_sum = sum(a)
    
    # We want (dist from s to t) % M == 0
    # Case 1: s < t
    # (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # Case 2: s > t
    # (Total_Sum - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total_Sum % M
    
    # Map all prefix sums to their remainders modulo M
    rems = list(map(lambda x: x % m, prefixes))
    counts = Counter(rems)
    
    # For Case 1 (s < t):
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If s < t and P_s % M == P_t % M, that's one pair.
    # If s > t and (P_s - P_t) % M == Total_Sum % M, that's another.
    
    # Let's redefine:
    # We seek pairs (s, t) with s != t such that:
    # If s < t: P_t % M == P_s % M
    # If s > t: P_t % M == (P_s - Total_Sum) % M
    
    # Let R_i = P_i % M.
    # Total pairs = sum_{s < t} [R_t == R_s] + sum_{s > t} [R_t == (R_s - Total_Sum) % M]
    
    # Part 1: s < t
    # This is simply the sum of c*(c-1)//2 for all counts c in Counter.
    ans1 = sum([c * (c - 1) // 2 for c in counts.values()])
    
    # Part 2: s > t
    # We need R_t == (R_s - Total_Sum) % M.
    # Let T = Total_Sum % M.
    # We need R_t == (R_s - T) % M, which is R_s == (R_t + T) % M.
    # For a fixed t, the number of s > t is the number of times (R_t + T) % M 
    # appears in the prefix sums at indices greater than t.
    # This is harder without loops. Let's use the property:
    # Total pairs = sum_{s, t (s!=t)} [dist(s, t) % M == 0]
    # dist(s, t) % M = (P_t - P_s) % M if s < t else (Total_Sum + P_t - P_s) % M
    
    # Let's use the observation:
    # For every pair {s, t} with s < t:
    # Clockwise s -> t is (P_t - P_s) % M
    # Clockwise t -> s is (Total_Sum + P_s - P_t) % M
    # Notice: (s -> t) + (t -> s) = Total_Sum % M
    
    # Let T = Total_Sum % M.
    # If T == 0:
    # Then (s -> t) % M == 0 if and only if (t -> s) % M == 0.
    # Each pair {s, t} with R_s == R_t contributes 2 to the answer.
    # Result = sum(c * (c - 1))
    
    # If T != 0:
    # Then (s -> t) % M == 0 and (t -> s) % M == 0 cannot both be true.
    # (s -> t) % M == 0  => R_t == R_s
    # (t -> s) % M == 0  => R_s == (R_t + T) % M
    # For each pair {s, t} with s < t:
    # It contributes 1 if R_s == R_t
    # It contributes 1 if R_s == (R_t + T) % M
    
    # Total = sum_{s < t} [R_s == R_t] + sum_{s < t} [R_s == (R_t + T) % M]
    # The first term is sum(c*(c-1)//2).
    # The second term: for each t, count s < t such that R_s == (R_t + T) % M.
    # This is equivalent to: for all s, t: count pairs (s, t) such that R_s == (R_t + T) % M
    # MINUS the cases where s >= t.
    # Actually, simpler:
    # The total number of pairs (s, t) with s != t such that dist(s, t) % M == 0 is:
    # sum_{r=0 to M-1} (count[r] * count[(r - T) % M])
    # BUT we must exclude cases where s == t.
    # If s == t, dist(s, t) is not defined (s != t), but the formula (P_t - P_s) % M 
    # would give 0. If T == 0, then (r - T) % M == r, so we subtract count[r] for each r.
    # If T != 0, then (r - T) % M != r, so no s == t cases are counted.
    
    t_rem = total_sum % m
    
    # Using a list comprehension to sum count[r] * count[(r - t_rem) % m]
    # We only iterate over the keys present in the counter to keep it efficient.
    total_pairs = sum([counts[r] * counts.get((r - t_rem) % m, 0) for r in counts])
    
    # Subtract cases where s == t (which happens if t_rem == 0)
    final_ans = total_pairs - (n if t_rem == 0 else 0)
    
    print(final_ans)

if __name__ == "__main__":
    solve()