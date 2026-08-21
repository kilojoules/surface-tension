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
    
    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A_1 % M, P[2] = (A_1 + A_2) % M, ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1]) % M
    
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # P has N+1 elements. P[0] is start, P[N] is total sum % M.
    # We only need P[0]...P[N-1] for the starting positions.
    P_reduced = P[:N]
    total_sum_mod = P[N]
    
    # Count occurrences of each remainder
    counts = Counter(P_reduced)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # Number of pairs is sum(count * (count - 1) // 2)
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # => P[s-1] - P[t-1] == total_sum_mod (mod M)
    # => P[s-1] - total_sum_mod == P[t-1] (mod M)
    
    # We need to count pairs (s, t) such that P[t-1] == (P[s-1] - total_sum_mod) % M
    # This is sum(counts[r] * counts[(r - total_sum_mod) % M])
    # However, we must exclude cases where s == t (though the problem says s != t)
    # and we must handle the case where total_sum_mod == 0 carefully.
    
    # Using a generator expression to sum the products
    ans_s_gt_t = sum(
        counts[r] * counts[(r - total_sum_mod) % M] 
        for r in counts
    )
    
    # If total_sum_mod == 0, the condition P[s-1] == P[t-1] is met.
    # The above loop counts all pairs (s, t) including s == t.
    # Since s > t, we only want pairs where s != t.
    # If total_sum_mod == 0, ans_s_gt_t currently includes sum(counts[r]^2).
    # We need to subtract the cases where s == t.
    if total_sum_mod == 0:
        ans_s_gt_t -= N
        
    # The logic for s > t is: for every s, how many t < s satisfy the condition?
    # The symmetry of the problem allows us to use the Counter.
    # If total_sum_mod == 0, the condition is P[s-1] == P[t-1].
    # There are sum(c*(c-1)) such pairs in total across both s < t and s > t.
    # If total_sum_mod != 0, the condition P[s-1] - P[t-1] == total_sum_mod
    # is distinct from P[t-1] - P[s-1] == total_sum_mod.
    
    # Correct approach for s > t:
    # We want pairs (s, t) such that 1 <= t < s <= N 
    # and (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # This is equivalent to P[t-1] == (P[s-1] - total_sum_mod) % M
    
    # Let's redefine:
    # For each remainder r, there are counts[r] indices.
    # For a fixed s, the number of t < s is not directly available from Counter.
    # But the total number of pairs (s, t) with s != t satisfying the condition is:
    # If total_sum_mod == 0: 
    #    Condition is P[s-1] == P[t-1]. Total pairs = sum(c * (c-1))
    # If total_sum_mod != 0:
    #    Condition is P[t-1] == (P[s-1] - total_sum_mod) % M.
    #    Total pairs = sum(counts[r] * counts[(r - total_sum_mod) % M])
    
    # Let's use the total logic:
    final_ans = 0
    if total_sum_mod == 0:
        final_ans = sum(c * (c - 1) for c in counts.values())
    else:
        # For every s, we need t such that P[t-1] = (P[s-1] - total_sum_mod) % M
        # This is simply the sum of counts[r] * counts[(r - total_sum_mod) % M]
        final_ans = sum(counts[r] * counts[(r - total_sum_mod) % M] for r in counts)
        
    print(final_ans)

if __name__ == "__main__":
    solve()