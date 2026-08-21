import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A_1 % M, P[2] = (A_1 + A_2) % M, ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1]) % M
    
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # P has N+1 elements. P[0] is start, P[N] is total sum % M.
    # We only need P[0]...P[N-1] for the starting positions.
    P_reduced = P[:N]
    total_sum_mod = P[N]
    
    # Count occurrences of each remainder modulo M
    counts = Counter(P_reduced)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # Number of pairs is sum(count * (count - 1) // 2)
    ans_s_less_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For a pair (s, t) with s > t:
    # Distance is (total_sum_mod - P[s-1] + P[t-1]) % M == 0
    # => P[s-1] - P[t-1] == total_sum_mod (mod M)
    # => P[s-1] - total_sum_mod == P[t-1] (mod M)
    # Let target = (P[s-1] - total_sum_mod) % M
    # We need to count pairs (s, t) such that P[t-1] == target
    
    # We can calculate this by iterating over the counts of P[s-1]
    # For each remainder 'r' that appears 'c' times:
    # The required P[t-1] is (r - total_sum_mod) % M
    # The number of pairs is c * counts[(r - total_sum_mod) % M]
    # However, we must exclude cases where s == t (though the problem says s != t)
    # and we must handle the case where total_sum_mod == 0 carefully.
    
    # If total_sum_mod == 0, then P[s-1] == P[t-1]. 
    # This is already covered by the s < t case if we just mirror it.
    # But the condition s > t is distinct.
    # If total_sum_mod == 0, then for every pair (s, t) with s < t and dist 0,
    # the pair (t, s) also has dist 0.
    
    # General formula for s > t:
    # sum(counts[r] * counts[(r - total_sum_mod) % M])
    # Then subtract cases where s == t (which happens if total_sum_mod == 0)
    
    ans_s_greater_t = sum(c * counts[(r - total_sum_mod) % M] 
                          for r, c in counts.items())
    
    # Subtract cases where s == t (only happens if total_sum_mod % M == 0)
    if total_sum_mod == 0:
        ans_s_greater_t -= N
        
    print(ans_s_less_t + ans_s_greater_t)

if __name__ == "__main__":
    solve()