import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A.
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M.
    # The distance from s to t (s > t) is (TotalSum - P_{s-1} + P_{t-1}) mod M.
    # We want distance % M == 0.
    
    # Calculate prefix sums modulo M. 
    # P[i] = sum(A[0...i-1]) % M. P[0] = 0.
    # We use accumulate to avoid loops.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of A modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # Note: P indices are 0 to N. The distance from s to t is sum(A[s-1...t-2]).
    # This is P[t-1] - P[s-1].
    # For a fixed value v, if it appears C_v times in P[0...N-1], 
    # there are C_v * (C_v - 1) // 2 pairs.
    
    # Case 2: s > t
    # (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # => P[s-1] - P[t-1] == total_sum_mod (mod M)
    
    # Let's use a Counter for P[0...N-1]
    counts = Counter(P[:N])
    
    # For Case 1: s < t, we need P[s-1] == P[t-1].
    # The number of pairs is sum(C_v * (C_v - 1) // 2)
    ans_s_lt_t = sum(v * (v - 1) // 2 for v in counts.values())
    
    # For Case 2: s > t, we need P[s-1] - P[t-1] == total_sum_mod (mod M).
    # This is equivalent to P[s-1] == (P[t-1] + total_sum_mod) (mod M).
    # For each t-1, we need to count how many s-1 (where s-1 > t-1) satisfy this.
    # However, it's easier to think: for every pair (t-1, s-1) with t-1 < s-1,
    # we check if P[s-1] - P[t-1] == total_sum_mod (mod M).
    # This is sum(count(v) * count((v + total_sum_mod) % M))
    # But we must exclude the case where s-1 == t-1 (which is impossible since s > t).
    # If total_sum_mod == 0, then P[s-1] == P[t-1], which is the same as Case 1.
    # But the problem says s != t.
    
    if total_sum_mod == 0:
        # If total_sum is 0 mod M, then s < t and s > t both reduce to P[s-1] == P[t-1].
        # Total pairs = N * (N-1) if all P are same, etc.
        # Actually, for each pair {i, j}, both (i, j) and (j, i) work.
        print(2 * ans_s_lt_t)
    else:
        # Case 2: s > t. We need P[s-1] - P[t-1] == total_sum_mod (mod M).
        # We can iterate over all v in counts and multiply by count of (v + total_sum_mod) % M.
        # Since total_sum_mod != 0, v != (v + total_sum_mod) % M, so no overlap with s=t.
        ans_s_gt_t = sum(counts[v] * counts.get((v + total_sum_mod) % M, 0) for v in counts)
        print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()