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
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Calculate prefix sums modulo M
    # accumulate provides P[1]...P[N]. We prepend 0 for P[0].
    prefix_sums = list(accumulate(A, lambda x, y: (x + y) % M))
    # P[0] is 0. The list prefix_sums currently contains P[1] to P[N].
    # Let's create the full list P = [0, P[1], ..., P[N-1]]
    # Note: P[N] is the total sum, which we need for the s > t case.
    total_sum_mod = prefix_sums[-1]
    P = [0] + prefix_sums[:-1]
    
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # And (total_sum_mod - P[s-1] + P[t-1]) % M == 0 for s > t
    
    # For s < t: P[t-1] % M == P[s-1] % M
    # For s > t: P[t-1] % M == (P[s-1] - total_sum_mod) % M
    
    # Count occurrences of each remainder in P
    counts = Counter(P)
    
    # Case 1: s < t
    # For each remainder r, if there are c occurrences, there are c*(c-1)//2 pairs.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need P[t-1] % M == (P[s-1] - total_sum_mod) % M
    # Let r_t = P[t-1] % M and r_s = P[s-1] % M.
    # r_t == (r_s - total_sum_mod) % M  =>  r_s == (r_t + total_sum_mod) % M
    # For each r_t, the number of pairs is counts[r_t] * counts[(r_t + total_sum_mod) % M].
    # However, this includes cases where s = t (which is forbidden) if total_sum_mod == 0.
    # But the problem says s != t. 
    # If total_sum_mod == 0, then r_s == r_t. The number of pairs (s, t) with s > t 
    # is the same as s < t, which is c*(c-1)//2.
    # If total_sum_mod != 0, then r_s != r_t, so s cannot be equal to t.
    
    if total_sum_mod == 0:
        ans_s_gt_t = ans_s_lt_t
    else:
        # Sum counts[r] * counts[(r + total_sum_mod) % M] for all r in counts
        ans_s_gt_t = sum(counts[r] * counts.get((r + total_sum_mod) % M, 0) for r in counts)
        
    print(ans_s_lt_t + ans_s_gt_t)

if __name__ == "__main__":
    solve()