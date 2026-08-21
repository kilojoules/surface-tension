import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums: P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A_1, P[2] = A_1 + A_2, ...
    # We use accumulate to avoid loops.
    prefix_sums = list(accumulate(a, lambda x, y: x + y, initial=0))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    # In both cases, we want (Distance) % M == 0.
    # This is equivalent to P[t-1] % M == P[s-1] % M if we consider the 
    # total sum is also a multiple of M, but the problem is simpler:
    # We want (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (Total_Sum + P[t-1] - P[s-1]) % M == 0 for s > t.
    
    # Let R[i] = P[i] % M.
    # For s < t: R[t-1] == R[s-1]
    # For s > t: R[t-1] == (R[s-1] - Total_Sum) % M
    
    remainders = [r % m for r in prefix_sums[:-1]]
    total_sum_rem = prefix_sums[-1] % m
    
    # Count occurrences of each remainder
    counts = Counter(remainders)
    
    # For each remainder r, there are counts[r] positions.
    # The number of pairs (s, t) with s < t such that R[s-1] == R[t-1] is:
    # sum(c * (c - 1) // 2 for c in counts.values())
    # However, we must also account for s > t.
    
    # Let's evaluate the condition: (P[t-1] - P[s-1]) % M == 0 for s < t
    # and (P[t-1] - P[s-1] + Total_Sum) % M == 0 for s > t.
    
    # If Total_Sum % M == 0, then both conditions are R[t-1] == R[s-1].
    # If Total_Sum % M != 0, the conditions are different.
    
    # Total pairs = sum_{r} (counts[r] * counts[r]) - N 
    # Wait, that's if Total_Sum % M == 0.
    
    # Correct logic:
    # For a fixed s, we need t such that:
    # 1. t > s and P[t-1] % M == P[s-1] % M
    # 2. t < s and P[t-1] % M == (P[s-1] - Total_Sum) % M
    
    # Let R_s = P[s-1] % M.
    # We need t > s with R_t = R_s  AND  t < s with R_t = (R_s - Total_Sum) % M.
    
    # Let's use the property:
    # Total pairs = sum_{s=1 to N} (count of t > s with R_t == R_s) 
    #              + sum_{s=1 to N} (count of t < s with R_t == (R_s - Total_Sum) % M)
    
    # This can be rewritten as:
    # sum_{r=0 to M-1} (counts[r] * (counts[r]-1) // 2)  <-- for s < t
    # + sum_{r=0 to M-1} (counts[r] * counts[(r - total_sum_rem) % m]) <-- for s > t
    # Note: in the second sum, if (r - total_sum_rem) % m == r, we must subtract 
    # the cases where s=t, but the condition s > t already handles that.
    # Actually, the second sum counts all t such that R_t == (R_s - Total_Sum) % M.
    # We only want t < s.
    
    # Let's use a different approach:
    # For every pair (s, t) with s != t, the clockwise distance is:
    # if s < t: P[t-1] - P[s-1]
    # if s > t: P[N] - P[s-1] + P[t-1]
    
    # We want:
    # s < t: P[t-1] % M == P[s-1] % M
    # s > t: P[t-1] % M == (P[s-1] - P[N]) % M
    
    # Let R_i = P[i] % M.
    # Ans = sum_{i < j} [R_i == R_j] + sum_{i > j} [R_j == (R_i - P[N]) % M]
    
    # The first term is sum(c*(c-1)//2 for c in counts.values())
    # The second term: for each i, we want to count j < i such that R_j == (R_i - P[N]) % M.
    # This is sum_{i=1 to N} (count of j < i with R_j == (R_i - P[N]) % M)
    
    # To compute the second term without loops, we can use a trick with 
    # a running count or just realize:
    # sum_{i > j} [R_j == (R_i - P[N]) % M] is the same as
    # sum_{r=0 to M-1} (counts[r] * counts[(r - total_sum_rem) % m])
    # MINUS the cases where i <= j.
    # But we only care about i > j.
    
    # Let's use the symmetry:
    # Let f(r1, r2) = count of i such that R_i == r1 and count of j such that R_j == r2.
    # The total number of pairs (i, j) with i != j such that 
    # (dist clockwise from i to j) % M == 0 is:
    # If Total_Sum % M == 0:
    #    Every pair (i, j) with R_i == R_j satisfies it.
    #    Ans = sum(c * (c - 1) for c in counts.values())
    # If Total_Sum % M != 0:
    #    For each i, there is exactly one r* = (R_i - Total_Sum) % M.
    #    We need j such that R_j == r*.
    #    Since Total_Sum % M != 0, r* != R_i.
    #    Thus i and j are automatically different.
    #    Ans = sum(counts[r] * counts[(r - total_sum_rem) % m] for r in counts)
    
    # Wait, if Total_Sum % M == 0, then (R_i - Total_Sum) % M == R_i.
    # The formula sum(counts[r] * counts[(r - total_sum_rem) % m]) 
    # becomes sum(counts[r] * counts[r]), which includes i=j.
    # So we subtract N.
    
    # Final logic:
    # If total_sum_rem == 0:
    #    ans = sum(c * (c - 1) for c in counts.values())
    # Else:
    #    ans = sum(counts[r] * counts[(r - total_sum_rem) % m] for r in counts)
    
    # We can unify this:
    # ans = sum(counts[r] * counts[(r - total_sum_rem) % m] for r in counts)
    # if total_sum_rem == 0: ans -= n
    
    # Using a generator expression to calculate the sum:
    total_pairs = sum(counts[r] * counts[(r - total_sum_rem) % m] for r in counts)
    
    # Subtract N if total_sum_rem is 0 because the condition R_j == (R_i - 0) % M 
    # is satisfied when i == j, but the problem says s != t.
    result = total_pairs - (n if total_sum_rem == 0 else 0)
    
    print(result)

if __name__ == "__main__":
    solve()