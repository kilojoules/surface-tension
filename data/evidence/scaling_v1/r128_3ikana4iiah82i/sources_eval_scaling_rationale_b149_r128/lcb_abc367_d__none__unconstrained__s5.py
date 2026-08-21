import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Using a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to consume the iterator
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Let P_i be the distance from rest area 1 to rest area i+1.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (clockwise) is:
    # If s < t: (P_t - P_s)
    # If s > t: (Total_Sum - P_s) + P_t
    
    # Calculate prefix sums modulo M
    # prefix_sums[i] is the distance from area 1 to area i+1
    prefix_sums = list(accumulate(a, lambda x, y: (x + y) % m, initial=0))
    
    # The total distance around the lake modulo M
    total_sum = prefix_sums[-1] # This is sum(A) % M
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # Let X = prefix_sums[s-1] and Y = prefix_sums[t-1].
    # If s < t: (Y - X) % M == 0  => Y % M == X % M
    # If s > t: (total_sum - X + Y) % M == 0 => (X - Y) % M == total_sum % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(prefix_sums[:-1])
    
    # For a fixed remainder r, let c = counts[r].
    # 1. Pairs (s, t) with s < t and X == Y:
    #    This is c * (c - 1) / 2 for each r.
    # 2. Pairs (s, t) with s > t and (X - Y) % M == total_sum:
    #    For each r, we need Y such that Y == (r - total_sum) % M.
    #    The number of pairs is counts[r] * counts[(r - total_sum) % m].
    #    Special case: if total_sum == 0, then (X - Y) % M == 0 is the same as X == Y.
    #    However, the problem says s != t.
    
    # To avoid loops, we use the keys of the Counter and a generator expression.
    
    # Part 1: s < t and X == Y
    # Note: The prefix_sums list has N elements (0 to N-1).
    # The number of pairs (s, t) with s < t and prefix_sums[s-1] == prefix_sums[t-1]
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: s > t and (X - Y) % M == total_sum
    # We need to sum counts[r] * counts[(r - total_sum) % m] for all r.
    # If total_sum == 0, this is sum(c * c), but we must exclude s == t.
    # Actually, if total_sum == 0, then (X - Y) % M == 0 is the same as X == Y.
    # The condition s > t is strictly different from s < t.
    
    # For s > t, we want (total_sum - X + Y) % M == 0
    # Y % M == (X - total_sum) % M
    ans_s_gt_t = sum(counts[r] * counts[(r - total_sum) % m] for r in counts)
    
    # If total_sum == 0, the s > t case includes pairs where X == Y.
    # The number of pairs (s, t) with s > t and X == Y is also c * (c - 1) // 2.
    # But the formula sum(counts[r] * counts[r]) includes s == t (which is c).
    # So we must subtract the diagonal.
    
    # Let's refine:
    # For each r, we have c = counts[r].
    # Pairs (s, t) with s < t and X == Y: c*(c-1)//2
    # Pairs (s, t) with s > t and X-Y == total_sum (mod M):
    #   If total_sum == 0: c*(c-1)//2
    #   If total_sum != 0: counts[r] * counts[(r - total_sum) % m]
    
    # Using a conditional in the generator to handle total_sum == 0
    final_ans = (
        sum(c * (c - 1) // 2 for c in counts.values()) + 
        (
            sum(c * (c - 1) // 2 for c in counts.values()) 
            if total_sum == 0 
            else sum(counts[r] * counts[(r - total_sum) % m] for r in counts)
        )
    )
    
    print(final_ans)

if __name__ == "__main__":
    solve()