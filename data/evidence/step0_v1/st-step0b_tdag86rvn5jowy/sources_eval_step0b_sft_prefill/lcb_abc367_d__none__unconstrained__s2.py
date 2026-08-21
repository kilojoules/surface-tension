```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # The distance from s to t (clockwise) is:
    # 1. If s < t: P_{t-1} - P_{s-1}
    # 2. If s > t: (P_N - P_{s-1}) + P_{t-1}
    # We want this distance to be 0 mod M.
    
    # Calculate prefix sums modulo M
    # We use a list comprehension to avoid explicit loops
    # P[i] = (sum of A[0...i-1]) % M
    P = list(accumulate([x % M for x in A], initial=0))
    # P now has N+1 elements: P[0], P[1], ..., P[N]
    # Note: P[N] is the total sum mod M.
    
    # Let x_i = P[i] for i = 0 to N-1.
    # These represent the positions of rest areas 1 to N relative to area 1.
    # For a pair (s, t):
    # If s < t: (x_{t-1} - x_{s-1}) % M == 0  => x_{t-1} == x_{s-1}
    # If s > t: (x_N - x_{s-1} + x_{t-1}) % M == 0 => (x_{s-1} - x_{t-1}) % M == x_N % M
    
    # Let's use a frequency map for x_i
    counts = Counter(P[:-1])
    total_sum_mod = P[-1]
    
    # For each unique value v in counts:
    # 1. Pairs (s, t) where s < t and x_{s-1} == x_{t-1} == v:
    #    If there are 'c' occurrences of v, there are c * (c - 1) // 2 pairs.
    #    However, the problem asks for (s, t), and since s < t, these are only clockwise.
    #    Wait, the problem says s != t. 
    #    If s < t, distance is P[t-1] - P[s-1].
    #    If s > t, distance is P[N] - P[s-1] + P[t-1].
    
    # Let's re-evaluate:
    # We want (P[t-1] - P[s-1]) % M == 0 when s < t
    # And (P[N] - P[s-1] + P[t-1]) % M == 0 when s > t
    
    # Let x_i = P[i] for i in 0...N-1.
    # We want pairs (i, j) with 0 <= i, j < N and i != j such that:
    # If i < j: (x_j - x_i) % M == 0  => x_i == x_j
    # If i > j: (x_N - x_i + x_j) % M == 0 => (x_i - x_j) % M == x_N % M
    
    # Let S = x_N % M.
    # Total = sum_{i < j} [x_i == x_j] + sum_{i > j} [(x_i - x_j) % M == S]
    
    # Part 1: sum_{i < j} [x_i == x_j]
    # For each unique value v with count c, this is c*(c-1)//2.
    part1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part 2: sum_{i > j} [(x_i - x_j) % M == S]
    # This is equivalent to sum_{i, j} [i > j and x_i - x_j \equiv S (mod M)]
    # We can iterate through the array x and for each x_i, count how many x_j (j < i) 
    # satisfy x_j \equiv (x_i - S) (mod M).
    
    # To do this without explicit loops, we can use a combination of map and list comprehensions,
    # but since we need the state of the counter to evolve, we can use a trick with 
    # a custom function or just use the mathematical property:
    # Total Part 2 = sum_{v} (count(v) * count((v - S) % M)) 
    # Subtract cases where i == j (which happens if S == 0) and handle the i > j constraint.
    
    # Actually, if we take all pairs (i, j) such that (x_i - x_j) % M == S:
    # If S == 0, we have pairs where x_i == x_j. There are c*(c-1) such pairs.
    # Half of them have i > j. So Part 1 + Part 2 = c*(c-1)//2 + c*(c-1)//2 = c*(c-1).
    # If S != 0, then x_i and x_j must be different.
    # For every pair {i, j}, one is larger. 
    # Either (x_i - x_j) % M == S or (x_j - x_i) % M == S.
    # So we just need to sum count(v) * count((v - S) % M) for all v.
    
    # Let's use the property:
    # For a fixed S, the number of pairs (i, j) with 0 <= i, j < N such that (x_i - x_j) % M == S is:
    # sum_{v} (count(v) * count((v - S) % M))
    
    # If S == 0:
    # The condition is x_i == x_j. 
    # For i < j, we have c(c-1)//2. For i > j, we have c(c-1)//2.
    # Total = c(c-1).
    
    # If S != 0:
    # The condition (x_i - x_j) % M == S is different from (x_j - x_i) % M == S.
    # For any two indices i, j, only one can satisfy this.
    # The total number of pairs (i, j) with i != j satisfying this is simply
    # sum_{v} (count(v) * count((v - S) % M)).
    
    S = total_sum_mod
    
    # Use a list comprehension to calculate the sum for Part 2 (and Part 1 if S==0)
    # We iterate over the unique keys of the counter.
    ans = sum(counts[v] * counts.get((v - S) % M, 0) for v in counts)
    
    # If S == 0, the above calculates sum(c*c). 
    # But we must exclude i == j, so it's sum(c*c - c) = sum(c*(c-1)).
    # This correctly covers both Part 1 and Part 2.
    # If S != 0, the above calculates sum(count(v) * count(v-S)).
    # Since S != 0, v != v-S, so i cannot be equal to j.
    # This correctly covers all pairs (i, j) regardless of whether i < j or i > j.
    
    # Wait, the logic for S != 0:
    # We want (i, j) such that:
    # (i < j and x_j - x_i == 0 mod M) OR (i > j and x_i - x_j == S mod M)
    # This is:
    # sum_{v} [c(v) * (c(v)-1) // 2] + sum_{i > j} [x_i - x_j == S mod M]
    
    # Let's use the property: 
    # sum_{i > j} [x_i - x_j == S mod M] + sum_{i < j} [x_i - x_j == S mod M] = sum_{v} [c(v) * c((v-S)%M)]
    # If S == 0, the two terms on the left are identical.
    # If S != 0, the two terms are different.
    
    # Let's use a different approach for S != 0:
    # We need sum_{i < j} [x_i == x_j] + sum_{i > j} [x_i - x_j == S mod M]
    # Let's process the array x and maintain counts of elements seen so far.
    
    # Since we can't use loops, we can use a combination of map/reduce or a trick.
    # But we can just use the fact that:
    # sum_{i > j} [x_i - x_j == S mod M] = sum_{i} (count of x_j == (x_i - S) mod M for j < i)
    
    # To do this without loops, we can use a custom reduce function.
    from functools import reduce
    
    def accumulate_counts(acc, val):
        # acc is (current_total, current_counts_dict)
        total, counts_dict = acc
        target = (val - S) % M
        new_total = total + counts_dict.get(target, 0)
        counts_dict[val] = counts_dict.get(val, 0) + 1
        return (new_total, counts_dict)

    # However, we also need Part 1: sum_{i < j} [x_i == x_j]
    # We can just add the condition [x_i == x_j] to the same reduce.
    
    def accumulate_both(acc, val):
        total, counts_dict = acc
        # Part 2: i > j and x_i - x_j == S mod M
        target_p2 = (val - S) % M
        # Part 1: i < j and x_i == x_j  =>  j > i and x_j == x_i
        # When we are at index j, we look for x_i == x_j for i < j.
        target_p1 = val
        
        new_total = total + counts_dict.get(target_p2, 0) + counts_dict.get(target_p1, 0)
        # If S == 0, target_p1 and target_p2 are the same, and we counted it twice.
        # But the problem says s != t. If S == 0, then x_i == x_j implies both conditions.
        # Actually, if S == 0, then (x_j - x_i) % M == 0 and (x_i - x_j) % M == 0.
        # The pair (s, t) is counted once if s < t and once if s > t.
        # This is correct.
        
        # Correction: if S == 0, target_p1 == target_p2. 
        # The logic `counts_dict.get(target_p2, 0) + counts_dict.get(target_p1, 0)` 
        # would double count.
        
        # Let's use a conditional:
        gain = counts_dict.get(target_p2, 0) if S != 0 else counts_dict.get(target_p1, 0)
        # Wait, if S == 0, then x_i == x_j is the only condition.
        # If s < t, dist = x_j - x_i. If s > t, dist = S + x_j - x_i.
        # If S = 0, both are x_j - x_i.
        # So for any pair {i, j}, we check if x_i == x_j. If yes, both (i, j) and (j, i) work.
        # That is 2 * c(c-1)//2 = c(c-1).
        
        # Let's use a simpler logic:
        # result = sum(c*(c-1)//2 for c in counts.values()) + sum(counts[v] * counts.get((v-S)%M, 0) for v in