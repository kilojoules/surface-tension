import sys
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    # We want (TotalSum - P[s-1] + P[t-1]) % M == 0
    
    total_sum_mod = P[-1]
    
    # We need to count pairs (s, t) such that:
    # 1. s < t and P[t-1] == P[s-1]
    # 2. s > t and (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    
    # Let's use a frequency map for P[0]...P[N-1]
    # Note: P[N] is the total sum, but the rest areas are 1...N.
    # The prefix sums for the starting points are P[0]...P[N-1].
    prefs = P[:-1]
    
    # Count occurrences of each remainder
    counts = {}
    for x in prefs:
        counts[x] = counts.get(x, 0) + 1
        
    # For a fixed remainder r, if there are c instances of r in prefs:
    # Pairs (s, t) with s < t and P[s-1] == P[t-1] == r: c * (c - 1) // 2
    # Pairs (s, t) with s > t and (total_sum_mod - P[s-1] + P[t-1]) % M == 0:
    # This is equivalent to P[s-1] == (P[t-1] + total_sum_mod) % M
    
    # Let r1 = P[t-1] and r2 = P[s-1].
    # Condition 1: r1 == r2 (where t > s)
    # Condition 2: r2 == (r1 + total_sum_mod) % M (where s > t)
    
    # Total pairs = sum_{r} (count(r) * count(r) - count(r)) / 2  <-- This is for s < t
    # Wait, the logic is simpler:
    # For every pair (s, t) with s != t:
    # If s < t, we need P[t-1] - P[s-1] = 0 mod M
    # If s > t, we need P[t-1] - P[s-1] + Total = 0 mod M => P[s-1] - P[t-1] = Total mod M
    
    # Let C(r) be the number of times remainder r appears in P[0]...P[N-1]
    # Pairs (s, t) with s < t: sum_{r} C(r)*(C(r)-1)//2
    # Pairs (s, t) with s > t: sum_{r} C(r) * C((r + total_sum_mod) % M)
    # BUT, if total_sum_mod == 0, the second condition becomes P[s-1] == P[t-1].
    # Since s > t, this is also C(r)*(C(r)-1)//2.
    
    # However, there is a catch: if total_sum_mod == 0, then P[s-1] == P[t-1] 
    # satisfies both s < t and s > t.
    # If total_sum_mod != 0, the sets of pairs are disjoint.
    
    # Correct logic:
    # For each r, we have C(r) indices.
    # 1. s < t: we need P[t-1] == P[s-1]. Total: sum(C(r)*(C(r)-1)//2)
    # 2. s > t: we need P[s-1] == (P[t-1] + total_sum_mod) % M.
    #    Total: sum(C(r) * C((r + total_sum_mod) % M))
    #    Wait, if total_sum_mod == 0, then P[s-1] == P[t-1]. 
    #    Since s > t, this is also sum(C(r)*(C(r)-1)//2).
    
    # Let's refine:
    # Let r_t = P[t-1] and r_s = P[s-1].
    # If s < t: valid if r_t == r_s.
    # If s > t: valid if (r_t - r_s + total_sum_mod) % M == 0  => r_s == (r_t + total_sum_mod) % M.
    
    # If total_sum_mod == 0:
    # Both conditions are r_s == r_t.
    # Total pairs = 2 * sum(C(r) * (C(r) - 1) // 2) = sum(C(r) * (C(r) - 1))
    
    # If total_sum_mod != 0:
    # Total pairs = sum(C(r) * (C(r) - 1) // 2) + sum(C(r) * C((r + total_sum_mod) % M))
    # Note: in the second sum, r is r_t. Since total_sum_mod != 0, (r + total_sum_mod) % M != r.
    # So we don't need to worry about s == t.
    
    c_vals = list(counts.values())
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in c_vals)
    
    if total_sum_mod == 0:
        print(2 * ans_s_lt_t)
    else:
        ans_s_gt_t = sum(counts[r] * counts.get((r + total_sum_mod) % M, 0) for r in counts)
        print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()