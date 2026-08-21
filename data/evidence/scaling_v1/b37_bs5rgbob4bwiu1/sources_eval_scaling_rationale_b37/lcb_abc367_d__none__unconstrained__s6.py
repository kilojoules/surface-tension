The constraint to avoid explicit loops forces a shift from imperative programming to a functional style. While not always "cleaner," it encourages the use of high-level abstractions and built-in optimizations rather than manual index management. For this problem, I will use `itertools.accumulate` to calculate prefix sums and `collections.Counter` to count occurrences of remainders, utilizing generator expressions to handle the data processing.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read N and M from the first line
    # Read A_i values from the remaining input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = map(int, input_data[2:])
    
    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2
    # ...
    # P_N = A_1 + ... + A_{N-1}
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t).
    
    # Calculate prefix sums P_i mod M
    # accumulate provides P_1, P_2, ..., P_N (where P_1 is 0)
    # We start with 0 and add A_1, A_2, ... A_{N-1}
    # To handle the wrap-around, we need the total sum of A
    
    # We use a list to store A to calculate total sum and prefix sums
    A_list = list(A)
    total_sum = sum(A_list)
    
    # P values: 0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}
    # We only need the first N prefix sums for the positions of the rest areas
    # The distance from s to t is (P_t - P_s) mod M = 0  => P_t = P_s (mod M)
    # This applies for s < t.
    # For s > t, distance is (Total - P_s + P_t) mod M = 0 => P_s - P_t = Total mod M
    
    # Generate P_i mod M for i = 1 to N
    # P_1 = 0, P_2 = A_1, ...
    # We take the first N elements of the accumulation of A_list starting with 0
    # Since accumulate(A_list) gives A_1, A_1+A_2..., we prepend 0.
    P = list(accumulate([0] + A_list))[:N]
    P_mod = [p % M for p in P]
    
    counts = Counter(P_mod)
    
    # For s < t: P_t % M == P_s % M
    # Number of pairs is sum(count * (count - 1) // 2)
    ans_st_lt = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For s > t: (Total + P_t - P_s) % M == 0  => (P_s - P_t) % M == Total % M
    # Let T = Total % M. We need P_s - P_t = T (mod M) => P_t = (P_s - T) % M
    # For every s, we need the number of t < s such that P_t = (P_s - T) % M.
    # However, it's easier to think: for every pair (s, t) with s > t, 
    # we check if (P_s - P_t) % M == T % M.
    # This is equivalent to counting pairs (P_s, P_t) such that P_s - P_t = T (mod M).
    # Total pairs (s, t) with s != t is N*(N-1).
    # The condition (dist from s to t) % M == 0 is:
    # If s < t: P_t - P_s = 0 (mod M)
    # If s > t: Total + P_t - P_s = 0 (mod M) => P_s - P_t = Total (mod M)
    
    T = total_sum % M
    
    # For s > t, we need P_s - P_t = T (mod M).
    # This means for each value v in P_mod, we look for (v - T) % M.
    # The number of such pairs is sum(count(v) * count((v - T) % M))
    # But we must exclude the case where s = t (which is already handled by s > t).
    # If T == 0, then P_s - P_t = 0 (mod M) is the same as the s < t case.
    # If T == 0, then for every pair {s, t}, both directions are multiples of M.
    # If T != 0, then s < t and s > t are distinct conditions.
    
    if T == 0:
        # Every pair {s, t} that satisfies the condition for s < t 
        # also satisfies it for s > t.
        print(ans_st_lt * 2)
    else:
        # Count pairs (s, t) with s > t such that P_s - P_t = T (mod M)
        # This is sum(counts[v] * counts[(v - T) % M] for v in counts)
        # Since s > t, we are looking for pairs of indices.
        # The total number of pairs (s, t) with s != t satisfying the condition is:
        # sum_{v} (count(v) * count((v - T) % M)) where T is the distance shift.
        # Wait, the simplest way:
        # For a fixed s and t, the clockwise distance is (P_t - P_s) mod Total.
        # We want (P_t - P_s) % M == 0 if s < t
        # and (Total + P_t - P_s) % M == 0 if s > t.
        
        # Let's use the property: 
        # Total pairs = sum_{v=0 to M-1} (count(v) * count((v + T) % M))
        # where T = Total % M. 
        # If T = 0, this is sum(count(v)^2), but we must subtract s=t cases: sum(count(v)).
        # If T != 0, this is sum(count(v) * count((v + T) % M)).
        
        # Let's re-evaluate:
        # Pair (s, t) is valid if:
        # 1. s < t and P_t - P_s = 0 mod M
        # 2. s > t and P_t - P_s = -Total mod M = (M - T) mod M
        
        # Let C_v be the number of i such that P_i = v mod M.
        # For a fixed v, there are C_v choices for P_s and C_v choices for P_t.
        # This gives C_v * (C_v - 1) pairs for s < t and s > t combined IF T=0.
        # If T != 0:
        # s < t: P_t = P_s mod M. Number of pairs: sum(C_v * (C_v - 1) / 2)
        # s > t: P_t = (P_s - T) mod M. Number of pairs: 
        # We need to count pairs (s, t) with s > t such that P_t = (P_s - T) mod M.
        # This is not simply C_v * C_{v-T} because of the s > t constraint.
        # Actually, the constraint s > t is handled by the fact that we are 
        # iterating over all pairs and splitting them into s < t and s > t.
        # For any two distinct indices i, j, one is smaller.
        # If i < j, distance is P_j - P_i. Valid if P_j - P_0 = P_i - P_0 mod M.
        # If i > j, distance is Total + P_j - P_i. Valid if P_j - P_i = -Total mod M.
        
        # Let's use the logic:
        # Total = sum_{i < j} [P_i == P_j mod M] + sum_{i > j} [P_i - P_j == Total mod M]
        # First term: sum(C_v * (C_v - 1) // 2)
        # Second term: sum_{i, j} [i > j and P_i - P_j == T mod M]
        # This second term is sum_{i} (count of j < i such that P_j = (P_i - T) mod M)
        
        # To calculate the second term without loops:
        # We can use a list comprehension and a running count, but that's a loop.
        # Wait, the second term is simply the number of pairs (i, j) with i > j 
        # such that P_i - P_j = T mod M.
        # This is NOT simply C_v * C_{v-T} because of the i > j condition.
        # Actually, it is! For any two indices i, j, if P_i - P_j = T mod M,
        # then either (i > j) or (i < j).
        # If i > j, it contributes to the "s > t" case.
        # If i < j, it contributes to the "s < t" case ONLY IF T = 0.
        # If T != 0, then P_i - P_j = T mod M and P_i - P_j = 0 mod M are mutually exclusive.
        # So for T != 0, the number of pairs (s, t) with s > t and P_s - P_t = T mod M
        # is simply the number of pairs (i, j) with i > j and P_i - P_j = T mod M.
        # This is still tricky without a loop. Let's use the fact that:
        # Total pairs (i, j) with i != j such that P_i - P_j = T mod M is sum(C_v * C_{(v-T)%M}).
        # For each such pair, either i > j or i < j.
        # If T != 0, then P_i - P_j = T mod M implies i != j.
        # Let X be the number of pairs (i, j) with i < j and P_i - P_j = T mod M.
        # Let Y be the number of pairs (i, j) with i > j and P_i - P_j = T mod M.
        # X + Y = sum(C_v * C_{(v-T)%M}).
        # We want Y. But we also know that X is the number of pairs i < j such that 
        # P_j - P_i = -T mod M = (M - T) mod M.
        # So X = sum_{i < j} [P_j - P_i = (M-T) mod M].
        # This is the same form as our "s < t" problem but with (M-T) instead of 0.
        # Actually, the most direct way to count i > j such that P_i - P_j = T mod M
        # is to iterate through the list and keep track of counts.
        # Since I can't use loops, I'll use a custom function with `map` or a generator
        # and `sum()`, but I need a state (the counter). 
        # I can use a dictionary and a generator expression with a side effect.
        
        # Correct logic for T != 0:
        # Ans = sum(C_v * (C_v - 1) // 2) + sum_{i=1..N} (count of j < i where P_j = (P_i - T) % M)
        
        # To implement the second term without a 'for' loop:
        # We can use a helper function and `sum()` with a generator.
        # We use a dictionary to track counts of P_j seen so far.
        
        state = Counter()
        def count_and_update(p):
            target = (p - T) % M
            res = state[target]
            state[p] += 1
            return res
            
        ans_st_gt = sum