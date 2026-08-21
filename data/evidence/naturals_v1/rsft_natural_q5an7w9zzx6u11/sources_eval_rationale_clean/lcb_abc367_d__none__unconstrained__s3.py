import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the remaining A_i
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension to capture the rest.
    it = input_data
    n = next(it)
    m = next(it)
    a = [x for x in it]

    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s)
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t)
    
    # Calculate prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + a[:-1]) gives us the distance from area 1 to area i
    # We only need the first N prefix sums because A_N is only used for s > t
    p = list(accumulate([0] + a[:-1]))
    total_sum = sum(a)
    
    # We want (dist(s, t)) % M == 0
    # Case 1: s < t
    # (P_t - P_s) % M == 0  =>  P_t % M == P_s % M
    # Case 2: s > t
    # (Total_Sum - P_s + P_t) % M == 0  =>  P_s % M == (Total_Sum + P_t) % M
    
    # Count frequencies of P_i % M
    counts = Counter([x % m for x in p])
    
    # For Case 1 (s < t):
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # If P_s % M == P_t % M, then (s, t) is a pair if s < t.
    # The total number of pairs (s, t) with s != t such that P_s % M == P_t % M 
    # is sum(c * (c - 1)). 
    # But wait, the condition for s > t is different.
    
    # Let's redefine:
    # We seek pairs (s, t) such that:
    # If s < t: P_t % M == P_s % M
    # If s > t: P_s % M == (Total_Sum + P_t) % M
    
    # Let R_i = P_i % M
    # Let T = Total_Sum % M
    # We want:
    # 1. s < t and R_t == R_s
    # 2. s > t and R_s == (T + R_t) % M
    
    # To calculate this without loops:
    # For a fixed t, we need count of s < t where R_s == R_t
    # PLUS count of s > t where R_s == (T + R_t) % M
    
    # Let's use the property:
    # Total pairs (s, t) with s != t is the sum over all t of:
    # (count of s < t with R_s == R_t) + (count of s > t with R_s == (T + R_t) % M)
    
    # This is equivalent to:
    # Sum_{t=1 to N} [ (count of s < t with R_s == R_t) + (count of s > t with R_s == (T + R_t) % M) ]
    
    # Let C(r) be the total count of i such that R_i == r.
    # The first term Sum_{t} (count s < t with R_s == R_t) is Sum_{r} C(r)*(C(r)-1)//2
    # The second term Sum_{t} (count s > t with R_s == (T + R_t) % M) is:
    # For each t, we need count of s in {t+1, ..., N} such that R_s == (T + R_t) % M.
    
    # Let's use a different approach for the second term:
    # It is Sum_{t=1 to N} Sum_{s=t+1 to N} [R_s == (T + R_t) % M]
    # This is Sum_{s=2 to N} Sum_{t=1 to s-1} [R_s == (T + R_t) % M]
    # Which is Sum_{s=2 to N} (count of t < s such that (T + R_t) % M == R_s)
    
    # Let's simplify:
    # We want pairs (s, t) such that:
    # If s < t, R_s == R_t
    # If s > t, R_s == (T + R_t) % M
    
    # Let's calculate:
    # Part A: Sum_{r=0 to M-1} C(r) * (C(r) - 1) // 2
    # Part B: Sum_{t=1 to N} (count of s > t such that R_s == (T + R_t) % M)
    
    # To calculate Part B without loops, we can use the total counts:
    # Sum_{t=1 to N} [ C((T + R_t) % M) - (count of s <= t such that R_s == (T + R_t) % M) ]
    
    # Actually, a simpler way:
    # Part A is the number of pairs (s, t) with s < t and R_s == R_t.
    # Part B is the number of pairs (s, t) with s > t and R_s == (T + R_t) % M.
    
    # Let's use the fact that:
    # Total pairs (s, t) with s != t such that (dist s to t) % M == 0 is:
    # Sum_{t=1 to N} (count s < t where R_s == R_t) + Sum_{t=1 to N} (count s > t where R_s == (T + R_t) % M)
    
    # Let's compute Part B by iterating through the list and keeping track of counts.
    # Since we can't use loops, we can use a trick with a custom function and 
    # a mutable object (like a dictionary) inside a list comprehension.
    
    # However, the simplest way to think about Part B:
    # It is Sum_{t=1 to N} [ (Total count of R_s == (T + R_t) % M) - (count of s <= t where R_s == (T + R_t) % M) ]
    
    # Let's use the "mutable state in list comprehension" pattern carefully.
    # We create a Counter for R and a running Counter for s <= t.
    
    r_values = [x % m for x in p]
    t_val = total_sum % m
    
    # Part A: s < t and R_s == R_t
    ans_a = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Part B: s > t and R_s == (T + R_t) % M
    # We need Sum_{t=1 to N} (Count of s > t such that R_s == (T + R_t) % M)
    # This is Sum_{t=1 to N} (counts[(T + R_t) % M] - (count of s <= t such that R_s == (T + R_t) % M))
    
    # To get "count of s <= t" without loops, we can use a generator with a side-effect
    # but that's frowned upon. Instead, let's use the fact that:
    # Sum_{t=1 to N} counts[(T + R_t) % M] is easy.
    # Sum_{t=1 to N} (count of s <= t such that R_s == (T + R_t) % M) 
    # is the sum over all t of the number of s in {1...t} such that R_s == (T + R_t) % M.
    
    # Let's use a helper function to track state.
    def get_running_counts(items):
        state = Counter()
        return [state.update({x: 1}) or state[ (t_val + x) % m ] for x in items]
    
    # The above is slightly wrong because it calculates count of s <= t where R_s == (T + R_t) % M
    # Let's refine:
    # Total B = Sum_{t=1 to N} counts[(T + R_t) % M] - Sum_{t=1 to N} (count s <= t where R_s == (T + R_t) % M)
    
    # To implement Sum_{t=1 to N} (count s <= t where R_s == (T + R_t) % M) without loops:
    # We can use a list comprehension that updates a dictionary.
    
    # Note: The constraint to avoid loops allows list comprehensions.
    # We can use a dictionary to store running counts.
    
    running_sum_b = 0
    # We need to calculate Sum_{t=1 to N} (count s <= t where R_s == (T + R_t) % M)
    # We can use a list comprehension and a dictionary.
    
    # Since we need to return a single integer, we can wrap the logic.
    # We'll use a list comprehension to drive the "loop" and a dictionary for state.
    
    state = Counter()
    # This list comprehension calculates the running count for each t
    # and stores it in a list, then we sum that list.
    # We use the 'or' trick to update the counter and return the value.
    
    # Correct logic for Part B:
    # For each t from 1 to N:
    #   1. Update state: state[R_t] += 1
    #   2. Add state[(T + R_t) % M] to running_total
    
    # But wait, the condition is s > t.
    # So for a fixed t, we want s in {t+1, ..., N}.
    # That is: (Total count of R_s == (T + R_t) % M) - (count of s <= t such that R_s == (T + R_t) % M)
    
    # Let's compute:
    # Term 1: Sum_{t=1 to N} counts[(T + R_t) % M]
    # Term 2: Sum_{t=1 to N} (count s <= t such that R_s == (T + R_t) % M)
    
    term1 = sum(counts[(t_val + r) % m] for r in r_values)
    
    # For Term 2, we use a list comprehension to simulate the loop
    # We use a dictionary to keep track of counts of R_s seen so far.
    # We use a list to store the results of the "side effect".
    
    # To avoid the 'or' trick which might be confusing, we can use a 
    # function that updates and returns.
    def update_and_get(r):
        state[r] += 1
        return state[(t_val + r) % m]
    
    term2 = sum([update_and_get(r) for r in r_values])
    
    print(ans_a + (term1 - term2))

if __name__ == "__main__":
    solve()