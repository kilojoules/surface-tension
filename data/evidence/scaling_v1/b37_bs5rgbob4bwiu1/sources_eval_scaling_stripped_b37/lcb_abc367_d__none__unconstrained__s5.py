```python
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
    
    # Let P_i be the prefix sum of distances from rest area 1.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_i = sum(A_j for j from 1 to i-1)
    # The distance from s to t (s < t) is (P_t - P_s) mod M == 0
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t) mod M == 0
    
    # Calculate prefix sums modulo M
    # accumulate([0] + A) gives [0, A_1, A_1+A_2, ...]
    # We only need the first N prefix sums (P_1 to P_N)
    prefixes = list(accumulate([0] + A, lambda x, y: (x + y) % M))[:N]
    
    # Total sum of all A_i modulo M
    total_sum_mod = sum(A) % M
    
    # Count occurrences of each remainder modulo M
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    for p in prefixes:
        counts[p] += 1
        
    # For a fixed s and t (s != t):
    # If s < t: (P_t - P_s) % M == 0  => P_t % M == P_s % M
    # If s > t: (Total - P_s + P_t) % M == 0 => (P_s - P_t) % M == Total % M
    
    # Case 1: s < t
    # For each remainder r, if there are 'c' indices with that remainder,
    # there are c*(c-1)//2 pairs (s, t) with s < t.
    # Total for all r: sum(c*(c-1)//2)
    
    # Case 2: s > t
    # We need (P_s - P_t) % M == total_sum_mod
    # This means P_t % M == (P_s - total_sum_mod) % M
    # For a fixed s, the number of t < s is the count of (P_s - total_sum_mod) % M
    # encountered so far. 
    # However, it's simpler to use the global counts:
    # For every s, we look for t such that P_t % M == (P_s - total_sum_mod) % M.
    # This includes t < s and t > s.
    # The condition s > t is specific. Let's use the property:
    # Total pairs = (Pairs where P_s == P_t) + (Pairs where P_s - P_t == Total)
    # Wait, the problem asks for pairs (s, t) where s != t.
    # Let's use the frequency map:
    # For each r1, let c1 = counts[r1].
    # These contribute c1*(c1-1) pairs where (P_t - P_s) % M == 0.
    # Wait, that's for both s < t and s > t? No.
    # If P_s == P_t, then (P_t - P_s) % M == 0. 
    # If s < t, this is the clockwise distance.
    # If s > t, the clockwise distance is (Total - P_s + P_t) % M.
    # If P_s == P_t, then (Total - P_s + P_t) % M == Total % M.
    
    # Correct Logic:
    # A pair (s, t) is valid if:
    # 1. s < t AND (P_t - P_s) % M == 0
    # 2. s > t AND (Total - P_s + P_t) % M == 0
    
    # Let's iterate through all possible remainders r in 0...M-1:
    # For a fixed r, let c = counts[r].
    # These c indices provide c*(c-1)//2 pairs for Case 1 (s < t).
    # For Case 2 (s > t), we need P_s - P_t \equiv Total (mod M).
    # Let r_s be the remainder of P_s and r_t be the remainder of P_t.
    # We need r_s - r_t \equiv Total (mod M) \Rightarrow r_t \equiv r_s - Total (mod M).
    # For every s, the number of t < s such that r_t == (r_s - Total) % M 
    # is the number of times (r_s - Total) % M appeared before s in the prefix list.
    
    # Let's use a list comprehension to calculate the answer:
    # Part 1: s < t and P_s == P_t
    # For each r, c = counts[r], contribution is c*(c-1)//2.
    # But we can't use loops, so we use a map/reduce or list comprehension.
    
    # Part 2: s > t and (Total - P_s + P_t) % M == 0
    # This is equivalent to P_t % M == (P_s - Total) % M.
    # For a fixed s, we need the count of t < s with remainder (P_s - Total) % M.
    # This is a prefix sum of counts.
    
    # Actually, there is a much simpler way:
    # For every pair (s, t) with s != t:
    # If s < t, condition is P_s \equiv P_t (mod M)
    # If s > t, condition is P_t \equiv P_s - Total (mod M)
    
    # Let's use the fact that we can iterate over the prefix list once:
    # For each P_s, it can be the 't' in (s < t) or the 's' in (s > t).
    # When we are at index i (which is P_i):
    # 1. It acts as 't': it pairs with all previous P_j == P_i.
    # 2. It acts as 's': it pairs with all previous P_j == (P_i - Total) % M.
    
    # We can use a custom function with reduce to maintain the state (counts_dict, total_pairs).
    # Since we cannot use loops, reduce is the way to go.
    
    def update_state(state, p):
        counts_dict, total_pairs = state
        # Pairs where current p is 't' (s < t): count of p seen so far
        # Pairs where current p is 's' (s > t): count of (p - total_sum_mod) % M seen so far
        s_less_t = counts_dict.get(p, 0)
        s_greater_t = counts_dict.get((p - total_sum_mod) % M, 0)
        
        # Update dictionary
        new_dict = counts_dict.copy()
        new_dict[p] = s_less_t + 1
        
        return (new_dict, total_pairs + s_less_t + s_greater_t)

    # To avoid the dictionary copy overhead and maintain O(N), 
    # we can use a list for counts and a custom object or a mutable container for the sum.
    # But we can't use loops. Let's use a list for counts and a list for the running total.
    
    # Wait, the most efficient way without loops:
    # Total = sum_{r=0}^{M-1} (counts[r] * counts[(r - total_sum_mod) % M])
    # This counts all pairs (s, t) such that P_t - P_s \equiv Total (mod M) if s > t
    # AND P_t - P_s \equiv 0 (mod M) if s < t.
    # Let's refine:
    # For a fixed pair {s, t} with s < t:
    # It is valid if P_s \equiv P_t (mod M) OR P_t - P_s \equiv -Total (mod M).
    # Note: -Total \equiv (M - Total) (mod M).
    # If Total \equiv 0 (mod M), then both conditions are the same: P_s \equiv P_t.
    # In that case, each pair {s, t} is counted twice (once as s < t, once as s > t).
    # If Total \not\equiv 0 (mod M), the two conditions are distinct.
    
    # Let's use the property:
    # Ans = \sum_{s < t} [P_s \equiv P_t] + \sum_{s > t} [P_t \equiv P_s - Total]
    # The second term is \sum_{t < s} [P_t \equiv P_s - Total]
    # This is exactly what the 'reduce' logic above does.
    # To make it efficient, we use a list for counts and a single-element list for the total.
    
    # Since we can't use loops, we can use a list comprehension to create a list of 
    # "current counts of (P_i - Total) % M" and "current counts of P_i".
    # But that requires knowing the counts at each step.
    
    # Let's use the mathematical approach:
    # For every r \in {0...M-1}:
    # Let c(r) be the number of i such that P_i \equiv r (mod M).
    # The number of pairs (s, t) with s < t and P_s \equiv P_t is \sum c(r)(c(r)-1)//2.
    # The number of pairs (s, t) with s > t and P_t \equiv P_s - Total is:
    # This is \sum_{s=1}^N (count of t < s such that P_t \equiv P_s - Total).
    # This is NOT simply a function of global counts unless we know the positions.
    # Actually, it is! 
    # \sum_{s=1}^N \sum_{t=1}^{s-1} [P_t \equiv P_s - Total]
    # Let r_s = P_s % M and r_t = P_t % M.
    # We want \sum_{s=1}^N \sum_{t=1}^{s-1} [r_t \equiv r_s - Total].
    # Let's use the identity:
    # \sum_{s < t} [r_s = r_t] + \sum_{s > t} [r_t = r_s - Total]
    # If Total \equiv 0 (mod M), then the second term is \sum_{s > t} [r_t = r_s],
    # which is the same as the first term. Total = 2 * \sum c(r)(c(r)-1)//2.
    # If Total \not\equiv 0 (mod M), the two conditions r_s = r_t and r_t = r_s - Total
    # are different.
    # The sum \sum_{s > t} [r_t = r_s - Total] is the number of pairs (t, s) with t < s
    # such that r_t = r_s - Total.
    # This is exactly the number of pairs (t, s) with t < s such that r_s - r_t = Total.
    # This is NOT simply c(r)*c(r-Total). That would be all pairs.
    # We only want t < s.
    
    # Wait, the problem can be solved by:
    # For every pair of indices i, j with i < j:
    # Pair (i, j) is valid if P_i \equiv P_j (mod M)
    # Pair (j, i) is valid if P_i \equiv P_j - Total (mod M)
    # Total valid = \sum_{i < j} ([P_i \equiv P_j] + [P_i \equiv P_j - Total])
    # = \sum_{i < j} [P_i \equiv P_j] + \sum_{i < j} [P_j \equiv P_i + Total]
    
    # Let