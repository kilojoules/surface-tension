import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)

    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We use accumulate to get prefix sums and take modulo M immediately
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M == 0
    # The distance from s to t (s > t) is (Total + P[t-1] - P[s-1]) % M == 0
    
    # Calculate prefix sums modulo M
    # P will have N elements: P[0]=0, P[1]=A_1%M, ..., P[N-1]=(A_1+...+A_{N-1})%M
    p = list(accumulate([0] + a[:-1], lambda x, y: (x + y) % m))
    
    # Total distance around the lake modulo M
    total_dist_m = sum(a) % m
    
    # Count occurrences of each remainder modulo M
    counts = Counter(p)
    
    # For a fixed s and t:
    # If s < t: we need P[t-1] % M == P[s-1] % M
    # If s > t: we need (total_dist_m + P[t-1] - P[s-1]) % M == 0
    #            which means P[s-1] % M == (total_dist_m + P[t-1]) % M
    
    # Case 1: s < t
    # For each remainder r, there are counts[r] positions. 
    # The number of pairs (s, t) with s < t is combinations(counts[r], 2)
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need P[s-1] % M == (total_dist_m + P[t-1]) % M
    # Let r_t = P[t-1] % M and r_s = P[s-1] % M.
    # We need r_s == (total_dist_m + r_t) % m.
    # For each possible remainder r, there are counts[r] choices for t
    # and counts[(total_dist_m + r) % m] choices for s.
    # However, we must exclude cases where s == t (though the problem says s != t,
    # the logic s > t already handles that, but we must ensure we don't 
    # count pairs where the condition is met but s is not actually > t).
    # Actually, the simplest way is:
    # For every pair (s, t) with s != t, the distance is either:
    # 1. P[t-1] - P[s-1] (if s < t)
    # 2. Total + P[t-1] - P[s-1] (if s > t)
    # These are distinct paths.
    
    # Let's use the property: 
    # Total pairs = Sum_{r} (counts[r] * counts[(total_dist_m + r) % m])
    # But we must subtract cases where s == t, which happens if total_dist_m == 0
    # because then r == (0 + r) % m.
    
    # Correct logic for s > t:
    # For each r, we have counts[r] options for t and counts[(total_dist_m + r) % m] for s.
    # This counts all pairs (s, t) such that dist(s, t) is a multiple of M and s > t
    # PROVIDED we handle the indices correctly.
    # Let's refine:
    # The number of pairs (s, t) with s < t such that P[t-1] - P[s-1] = 0 mod M
    # is sum(c*(c-1)//2).
    # The number of pairs (s, t) with s > t such that Total + P[t-1] - P[s-1] = 0 mod M
    # is sum(counts[r] * counts[(total_dist_m + r) % m]) 
    # MINUS the cases where s == t (which happens if total_dist_m == 0).
    # Wait, if total_dist_m == 0, then s > t and s < t are both covered by P[s] == P[t].
    # If total_dist_m == 0, then dist(s, t) is a multiple of M iff P[s-1] == P[t-1].
    # There are N(N-1) such pairs if we don't care about s < t or s > t? 
    # No, if total_dist_m == 0, then for any two indices i, j with P[i]==P[j],
    # both clockwise(i, j) and clockwise(j, i) are multiples of M.
    # So it's sum(c * (c-1)) for all c in counts.
    
    # If total_dist_m != 0:
    # s < t: P[t-1] == P[s-1]  => sum(c*(c-1)//2)
    # s > t: P[s-1] == (total_dist_m + P[t-1]) % m => sum(counts[r] * counts[(total_dist_m + r) % m])
    # Note: if total_dist_m != 0, then r != (total_dist_m + r) % m, so s cannot be t.
    
    if total_dist_m == 0:
        print(sum(c * (c - 1) for c in counts.values()))
    else:
        ans_s_gt_t = sum(counts[r] * counts.get((total_dist_m + r) % m, 0) for r in counts)
        print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()