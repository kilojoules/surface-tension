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
    # P[1] = A[0]
    # P[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t-1] - P[s-1].
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1].
    
    # Calculate prefix sums
    # P will have N elements: P[0] is distance from 1 to 1 (0), P[N-1] is distance from 1 to N.
    P = list(accumulate(A[:-1], initial=0))
    total_sum = sum(A)
    
    # We want (dist(s, t)) % M == 0.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M.
    # For s > t: (total_sum - P[s-1] + P[t-1]) % M == 0 => (P[t-1] - P[s-1]) % M == -total_sum % M.
    
    # Remainders of prefix sums
    rems = [p % M for p in P]
    counts = Counter(rems)
    
    # Case 1: s < t
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need (P[t-1] - P[s-1]) % M == (-total_sum) % M.
    # Let target = (-total_sum) % M.
    # We need P[t-1] % M == (P[s-1] + target) % M.
    target = (-total_sum) % M
    
    # For each s (represented by P[s-1]), we need to count how many t < s satisfy the condition.
    # However, it is easier to iterate over all possible remainders r1 (for s) and r2 (for t).
    # The condition is r2 == (r1 + target) % M.
    # The number of pairs (s, t) with s > t is sum(count(r1) * count(r2)) 
    # BUT we must exclude the case where s=t (which is already handled by the problem statement s != t).
    # Actually, the simplest way to calculate s > t is:
    # For every pair (s, t) with s != t, either s < t or s > t.
    # The total number of pairs (s, t) such that dist(s, t) % M == 0 is:
    # Sum over r: count(r) * count((r + total_sum) % M)
    # Then we subtract the cases where s == t (which is N cases if total_sum % M == 0).
    
    # Let's use the logic: 
    # For a fixed s, we need t such that dist(s, t) % M == 0.
    # If s < t, P[t-1] % M == P[s-1] % M.
    # If s > t, P[t-1] % M == (P[s-1] - total_sum) % M.
    
    # Total pairs = Sum_{r} (count(r) * count((r - total_sum) % M))
    # This includes s=t if (0 - total_sum) % M == 0, i.e., total_sum % M == 0.
    
    # Let's refine:
    # For each s in {1...N}, let r_s = P[s-1] % M.
    # We seek t in {1...N}, t != s, such that:
    # 1. t > s and r_t == r_s
    # 2. t < s and r_t == (r_s - total_sum) % M
    
    # This is equivalent to:
    # Sum_{s=1 to N} [ (count(r_s) - 1) if t > s is counted ] ... this is confusing.
    
    # Let's use:
    # Total = Sum_{r} (count(r) * count((r - total_sum) % M))
    # If total_sum % M == 0, then (r - total_sum) % M == r, so we get Sum count(r)^2.
    # In that case, we subtract N because s=t is counted.
    # If total_sum % M != 0, then r != (r - total_sum) % M, so s=t is never counted.
    
    total_pairs = sum(counts[r] * counts[(r - total_sum) % M] for r in counts)
    
    if total_sum % M == 0:
        print(total_pairs - N)
    else:
        print(total_pairs)

if __name__ == "__main__":
    solve()