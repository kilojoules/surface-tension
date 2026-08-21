The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state changes. This approach often leads to more concise code by utilizing Python's powerful built-in higher-order functions and comprehensions, which are highly optimized. To solve this problem, I will use `itertools.accumulate` to calculate prefix sums of the distances, list comprehensions for mapping these sums to their remainders modulo $M$, and `collections.Counter` to efficiently count occurrences of each remainder. The final answer is derived by summing the combinations of pairs sharing the same remainder, adjusted for the circular nature of the path.

```python
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
    
    # Prefix sums of distances: P[i] is distance from area 1 to area i+1
    # P = [0, A1, A1+A2, ..., A1+...+AN]
    # We only need the first N prefix sums (0 to N-1)
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1] + P[t-1])
    
    # Calculate prefix sums modulo M
    # P_mod[i] = (sum(A[0...i-1])) % M
    P_mod = list(map(lambda x: x % M, accumulate(A, initial=0)))[:-1]
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder
    counts = Counter(P_mod)
    
    # For a pair (s, t) with s < t:
    # Distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # For a pair (s, t) with s > t:
    # Distance is (Total - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == Total % M
    
    # Case 1: s < t
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs
    ans_s_lt_t = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need (P[s-1] - P[t-1]) % M == total_sum_mod
    # Which means P[t-1] % M == (P[s-1] - total_sum_mod) % M
    # For each s, we need the count of t < s such that P[t-1] % M == (P[s-1] - total_sum_mod) % M
    # However, it's simpler to iterate over all possible remainders r1:
    # Pair (s, t) where s > t is valid if (P[s-1] - P[t-1]) % M == total_sum_mod
    # Let r1 = P[t-1] % M and r2 = P[s-1] % M
    # We need (r2 - r1) % M == total_sum_mod  => r1 = (r2 - total_sum_mod) % M
    
    # To avoid loops, we use a generator expression inside sum()
    # We iterate over the unique remainders present in the Counter
    ans_s_gt_t = sum(
        counts[r] * counts[(r - total_sum_mod) % M]
        for r in counts
    )
    
    # The above logic for s > t counts pairs (s, t) where s > t.
    # But if total_sum_mod == 0, then r == (r - total_sum_mod) % M,
    # and we are counting pairs (s, t) where s > t and P[s-1] == P[t-1].
    # If total_sum_mod == 0, the condition (P[s-1] - P[t-1]) % M == 0 is the same as Case 1.
    # If total_sum_mod != 0, the sets of pairs are disjoint.
    
    # Correct logic for s > t:
    # For every s in {1..N}, we need t < s such that P[t-1] % M == (P[s-1] - total_sum_mod) % M.
    # This is tricky without a loop because the count of t depends on the index s.
    # Actually, the total number of pairs (s, t) with s != t is:
    # Sum over all r: counts[r] * counts[(r - total_sum_mod) % M]
    # BUT, this includes cases where s < t AND s > t if we aren't careful.
    
    # Let's redefine:
    # A pair (s, t) is valid if (P[t-1] - P[s-1]) % M == 0 (for s < t)
    # OR (Total + P[t-1] - P[s-1]) % M == 0 (for s > t)
    
    # Let's use the property: 
    # For a fixed s and t, the clockwise distance is:
    # If s < t: Dist = P[t-1] - P[s-1]
    # If s > t: Dist = (P[N-1] - P[s-1]) + P[t-1] = Total - P[s-1] + P[t-1]
    
    # Let's use a different approach to avoid loops and recursion:
    # We want pairs (s, t) such that Dist(s, t) % M == 0.
    # This is equivalent to:
    # For s < t: P[t-1] % M == P[s-1] % M
    # For s > t: P[t-1] % M == (P[s-1] - Total) % M
    
    # Let's calculate this using the Counter:
    # For every pair of indices (i, j) with 0 <= i < j < N:
    # i corresponds to s, j corresponds to t: valid if P[j] % M == P[i] % M
    # j corresponds to s, i corresponds to t: valid if P[i] % M == (P[j] - Total) % M
    
    # Total = sum_{i < j} [P[j]%M == P[i]%M] + sum_{i < j} [P[i]%M == (P[j]-Total)%M]
    # First term: sum(c*(c-1)//2 for c in counts.values())
    # Second term: This requires knowing the distribution of P[i] and P[j] relative to each other.
    # Wait, the second term is: for each j, count i < j such that P[i]%M == (P[j]-Total)%M.
    # This can be solved by iterating through the list and keeping a running count.
    # Since loops are forbidden, we can use a custom function with functools.reduce.
    
    from functools import reduce
    
    def accumulate_counts(acc, x):
        current_count, total_valid = acc
        target = (x - total_sum_mod) % M
        # Update total_valid by adding how many times 'target' has appeared so far
        # Update current_count by incrementing the count of the current remainder x
        new_count = current_count.copy()
        new_count[x] = new_count.get(x, 0) + 1
        return (new_count, total_valid + current_count.get(target, 0))

    # We use reduce to simulate the loop and maintain state
    final_state = reduce(accumulate_counts, P_mod, ({}, 0))
    
    # The first term (s < t) is handled by the logic: P[j]%M == P[i]%M
    # The second term (s > t) is handled by the reduce logic.
    # But the reduce logic as written handles s > t (where j is s and i is t).
    # We need to add the s < t cases.
    
    # Let's refine:
    # For i < j:
    # Pair (s=i+1, t=j+1) is valid if P[j] % M == P[i] % M
    # Pair (s=j+1, t=i+1) is valid if P[i] % M == (P[j] - Total) % M
    
    # Let's use reduce to calculate both in one pass:
    def solve_final(acc, x):
        counts, total_valid = acc
        # s < t: current x is t, we need i < j such that P[i] == x
        # s > t: current x is s, we need i < j such that P[i] == (x - Total) % M
        c_same = counts.get(x, 0)
        c_diff = counts.get((x - total_sum_mod) % M, 0)
        
        new_counts = counts.copy()
        new_counts[x] = c_same + 1
        return (new_counts, total_valid + c_same + c_diff)

    # Special case: if Total % M == 0, then (x - Total) % M == x.
    # The logic above counts the same pair twice if Total % M == 0.
    # However, the problem says s != t.
    # If Total % M == 0, then Dist(s, t) % M == 0 iff Dist(t, s) % M == 0.
    # The reduce logic:
    # For i < j:
    # if P[i] == P[j], then (s=i+1, t=j+1) is valid AND (s=j+1, t=i+1) is valid.
    # The reduce adds c_same (for s < t) and c_diff (for s > t).
    # If Total % M == 0, c_same == c_diff, so it adds 2 * c_same. Correct.
    # If Total % M != 0, c_same and c_diff are different. Correct.
    
    # One edge case: if Total % M == 0, and P[i] == P[j], then both are valid.
    # If Total % M != 0, and P[i] == P[j], only (s=i+1, t=j+1) is valid.
    # If Total % M != 0, and P[i] == (P[j] - Total) % M, only (s=j+1, t=i+1) is valid.
    
    # The only issue is when Total % M == 0 AND P[i] == P[j].
    # Then (x - Total) % M == x, so c_same == c_diff.
    # The reduce adds c_same + c_same = 2 * c_same.
    # This is correct because both (s=i+1, t=j+1) and (s=j+1, t=i+1) are valid.
    
    # Wait, there is one more case: what if Total % M == 0 and we are looking for s > t?
    # The distance is (Total - P[s-1] + P[t-1]) % M.
    # If Total % M == 0, this is (-P[s-1] + P[t-1]) % M.
    # For this to be 0, P[s-1] % M == P[t-1] % M.
    # This is exactly what the reduce logic does.
    
    # Final check on Total % M == 0:
    # If P[i] = 1, P[j] = 1, Total = 0 (mod M)
    # s=i+1, t=j+1: Dist = 1 - 1 = 0 (Valid)
    # s=j+1, t=i+1: Dist = 0 - 1 + 1 = 0 (Valid)
    # Reduce: c_same = 1, c_diff = 1. Total_valid += 2. Correct.
    
    # If Total % M != 0:
    # If P[i] = 1, P[j] = 1, Total =