import sys
from itertools import accumulate
from functools import reduce

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate prefix sums of A modulo M.
    # P[i] is the distance from rest area 1 to rest area i+1.
    # P = [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    # We use accumulate to avoid loops.
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M.
    # This is 0 mod M if P[t-1] == P[s-1].
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M.
    # This is 0 mod M if (P[s-1] - P[t-1]) % M == TotalSum % M.
    
    total_sum_mod = P[-1]
    
    # We only care about P[0]...P[N-1] for the starting/ending points.
    # P[N] is the total sum, which is used for the wrap-around logic.
    coords = P[:-1]
    
    # Count occurrences of each remainder modulo M.
    # Using a list as a frequency array since M <= 10^6.
    counts = [0] * M
    for x in coords:
        counts[x] += 1
    
    # For each remainder r, let c = counts[r].
    # 1. Pairs (s, t) where s < t and P[s-1] == P[t-1]:
    #    There are c * (c - 1) / 2 such pairs.
    # 2. Pairs (s, t) where s > t and (P[s-1] - P[t-1]) % M == total_sum_mod:
    #    This is P[s-1] - total_sum_mod == P[t-1] (mod M).
    #    For a fixed r = P[s-1], we need P[t-1] = (r - total_sum_mod) % M.
    #    There are counts[r] * counts[(r - total_sum_mod) % M] such pairs.
    #    Special case: if total_sum_mod == 0, then r == (r - 0) % M.
    #    The s > t condition is already handled by the logic, but we must
    #    be careful not to double count s < t and s > t if total_sum_mod == 0.
    
    # Calculate sum of c*(c-1)//2 for all r
    internal_pairs = sum(c * (c - 1) // 2 for c in counts)
    
    # Calculate sum of counts[r] * counts[(r - total_sum_mod) % M]
    # If total_sum_mod == 0, the condition (P[s-1] - P[t-1]) % M == 0 
    # is the same as P[s-1] == P[t-1]. 
    # For a fixed remainder r with count c, there are c*(c-1) total pairs (s,t) with s != t.
    # If total_sum_mod != 0, the s < t and s > t cases are disjoint.
    
    if total_sum_mod == 0:
        # Every pair (s, t) with P[s-1] == P[t-1] works regardless of order.
        # Total is sum(c * (c - 1))
        ans = sum(c * (c - 1) for c in counts)
    else:
        # s < t: P[s-1] == P[t-1]
        # s > t: P[s-1] - P[t-1] == total_sum_mod (mod M)
        external_pairs = sum(counts[r] * counts[(r - total_sum_mod) % M] for r in range(M))
        ans = internal_pairs * 2 + external_pairs
        # Wait, the internal_pairs logic was c(c-1)//2. 
        # Let's redefine:
        # Pairs (s, t) with s < t: P[s-1] == P[t-1] -> sum(c*(c-1)//2)
        # Pairs (s, t) with s > t: P[s-1] - P[t-1] == total_sum_mod (mod M) -> sum(counts[r] * counts[(r-total_sum_mod)%M])
        # But if total_sum_mod == 0, the second condition is also P[s-1] == P[t-1].
        # Since s > t, for each pair {s, t} there is exactly one such pair.
        # So it's sum(c*(c-1)//2) + sum(c*(c-1)//2) = sum(c*(c-1)).
        
        # Correct logic for total_sum_mod != 0:
        # Ans = sum(c*(c-1)//2 for c in counts) + sum(counts[r] * counts[(r-total_sum_mod)%M] for r in range(M))
        # However, the second term counts pairs (s, t) where s > t.
        # Let's re-verify: 
        # If s > t, dist = (Total - P[s-1]) + P[t-1] = Total - (P[s-1] - P[t-1])
        # For this to be 0 mod M: P[s-1] - P[t-1] = Total mod M.
        # This is exactly what `external_pairs` calculates.
        
        ans = sum(c * (c - 1) // 2 for c in counts) + \
              sum(counts[r] * counts[(r - total_sum_mod) % M] for r in range(M))

    print(ans)

if __name__ == "__main__":
    solve()