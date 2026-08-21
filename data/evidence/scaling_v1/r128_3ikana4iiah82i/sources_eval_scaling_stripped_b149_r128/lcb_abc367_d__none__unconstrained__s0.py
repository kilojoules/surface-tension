```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate the prefix sums of distances modulo M.
    # P[i] is the distance from rest area 1 to rest area i+1.
    # P[0] = 0 (distance from 1 to 1)
    # P[1] = A[0] % M (distance from 1 to 2)
    # ...
    # P[N] = sum(A) % M (distance from 1 to 1, completing the circle)
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M.
    # This is 0 mod M if P[t-1] == P[s-1].
    # The distance from s to t (s > t) is (P[N] - P[s-1] + P[t-1]) % M.
    # This is 0 mod M if P[s-1] - P[t-1] == P[N] % M.

    # Let total_sum = P[N].
    # We are looking for pairs (s, t) with 1 <= s, t <= N and s != t such that:
    # If s < t: P[t-1] - P[s-1] \equiv 0 (mod M)  => P[t-1] == P[s-1]
    # If s > t: P[N] - P[s-1] + P[t-1] \equiv 0 (mod M) => P[s-1] - P[t-1] == P[N]
    
    # We only care about P[0]...P[N-1] for the starting and ending points.
    # Let's use a Counter to store frequencies of P[0...N-1].
    counts = Counter(P[:N])
    total_sum = P[N]
    
    # For s < t, the number of pairs is sum(count * (count - 1) // 2) for each unique value in P.
    # However, the problem asks for pairs (s, t), and the condition s < t is just one case.
    # Let's evaluate the two conditions:
    # 1. s < t and P[t-1] == P[s-1]
    # 2. s > t and P[s-1] - P[t-1] == total_sum (mod M)
    
    # For condition 1:
    # For each value v that appears C times in P[0...N-1], there are C*(C-1)//2 pairs.
    ans1 = sum(C * (C - 1) // 2 for C in counts.values())
    
    # For condition 2:
    # We need P[s-1] - P[t-1] == total_sum (mod M) with s > t.
    # This is equivalent to P[s-1] - total_sum == P[t-1] (mod M).
    # Let's iterate through the unique values in counts.
    # For a value v, if we treat it as P[s-1], we need P[t-1] = (v - total_sum) % M.
    # Since s > t, we can't simply multiply counts. We need to track indices.
    # Wait, the condition s > t is naturally handled if we just look for pairs (t, s) 
    # such that t < s and P[s-1] - P[t-1] == total_sum (mod M).
    
    # Let's redefine: 
    # We want pairs (s, t) such that:
    # (s < t AND P[t-1] == P[s-1]) OR (s > t AND P[s-1] - P[t-1] == total_sum % M)
    
    # For the second part:
    # We want pairs (t, s) with t < s such that P[s-1] - P[t-1] == total_sum % M.
    # This is P[s-1] - total_sum == P[t-1] (mod M).
    # Let target(v) = (v - total_sum) % M.
    # We want to count pairs (t, s) with t < s such that P[t-1] == target(P[s-1]).
    
    # To do this without loops, we can use a technique to count inversions/pairs.
    # But since we can't use loops, we can use the fact that:
    # Total pairs (t, s) with t < s such that P[s-1] - P[t-1] == total_sum % M
    # is equivalent to counting how many times each value appears and using a 
    # mathematical approach if total_sum is 0.
    # If total_sum == 0, then condition 2 is P[s-1] == P[t-1] with s > t.
    # This is again C*(C-1)//2 for each value.
    
    # If total_sum != 0:
    # We need to count pairs (t, s) with t < s such that P[t-1] == (P[s-1] - total_sum) % M.
    # This is tricky without loops. Let's use a different approach.
    # We can use a list comprehension to build a list of "matches" and then sum them.
    # But we can't iterate. We can use a dictionary/Counter and a generator.
    
    # Let's use the property: 
    # Total pairs (s, t) with s != t such that dist(s, t) == 0 mod M.
    # dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # dist(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    
    # Notice that (P[N] - P[s-1] + P[t-1]) % M == 0 
    # is equivalent to (P[t-1] - P[s-1]) % M == -P[N] % M.
    
    # Let target = (-total_sum) % M.
    # We want:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (P[t-1] - P[s-1]) % M == target
    
    # If target == 0, then both conditions are (P[t-1] - P[s-1]) % M == 0.
    # For any two indices i, j with P[i] == P[j], one will be s < t and one s > t.
    # So it's just C * (C - 1) for each value.
    
    # If target != 0:
    # Condition 1: P[t-1] == P[s-1] (s < t) -> C * (C - 1) // 2
    # Condition 2: P[t-1] - P[s-1] == target (mod M) (s > t)
    # This is P[s-1] == (P[t-1] - target) % M (s > t)
    
    # Let's use a more general approach:
    # For every pair of indices i < j:
    # Check if P[j] - P[i] == 0 (mod M)  --> (s=i+1, t=j+1)
    # Check if P[j] - P[i] == -total_sum (mod M) --> (s=j+1, t=i+1)
    
    # Let target = (-total_sum) % M.
    # We want to count pairs i < j such that:
    # P[j] - P[i] == 0 (mod M) OR P[j] - P[i] == target (mod M).
    # Note: if target == 0, these are the same.
    
    # To count i < j such that P[j] - P[i] == K (mod M):
    # This is P[j] - K == P[i] (mod M).
    # We can iterate through the array and keep track of counts of P[i] seen so far.
    # But we can't use loops. We can use a custom function with `reduce`.
    
    from functools import reduce
    
    def count_pairs(K):
        # Counts pairs i < j such that P[j] - P[i] == K (mod M)
        # state: (current_count, running_counter_dict)
        def step(state, val):
            cur_cnt, run_cnt = state
            # We need P[i] == (val - K) % M
            needed = (val - K) % M
            return (cur_cnt + run_cnt.get(needed, 0), 
                    run_cnt.update({val: run_cnt.get(val, 0) + 1}) or run_cnt)
        
        # Since we can't use loops, we use a trick to update the dict in the return
        # However, dict.update returns None. We can use a helper to return the dict.
        def update_and_return(d, k, v):
            d[k] = d.get(k, 0) + v
            return d

        def step_fixed(state, val):
            cur_cnt, run_cnt = state
            needed = (val - K) % M
            return (cur_cnt + run_cnt.get(needed, 0), update_and_return(run_cnt, val, 1))

        return reduce(step_fixed, P[:N], (0, {}))[0]

    # The answer is count_pairs(0) + count_pairs(target)
    # But if target == 0, we only call it once.
    
    target = (-total_sum) % M
    result = count_pairs(0) + (count_pairs(target) if target != 0 else 0)
    print(result)

# To avoid the function definition inside solve and maintain the "no loop" constraint
# while using reduce, I'll wrap the logic.
# Actually, the most concise way to write this is to use a generator expression 
# inside sum() and a dictionary for the counts, but that requires a loop.
# The only way to "loop" without 'for' or 'while' is 'reduce', 'map', 'filter', 
# or recursion.

# Let's refine the reduce logic to be perfectly compliant.
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, M = int(input_data[0]), int(input_data[1])
    A = list(map(int, input_data[2:]))
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    total_sum = P[N]
    target = (-total_sum) % M
    
    def get_count(K):
        def step(state, val):
            cnt, d = state
            # Update dictionary and add to count
            # We use a list for the dictionary to mutate it inside a lambda
            # but since we need to return it, we just use a helper.
            return (cnt + d.get((val - K) % M, 0), 
                    (d.update({val: d.get(val, 0) + 1}) or d))
        
        # We need to bypass the 'or d' since update returns None
        # Let's use a more robust way to update the dict.
        # We can use a class or a mutable object.
        pass

# Since I cannot use loops, I will use a dictionary and a list comprehension 
# to simulate the counting, but that's essentially a loop.
# The only way to truly avoid 'for' is to use 'reduce'.
# Let's use a helper function for the dictionary update.

def helper_update(d, k, v):
    d[k] = d.get(k, 0) + v
    return d

def run_reduce(P, N, M, K):
    return reduce(lambda state, val: (state[0] + state[1].get