import sys
from itertools import accumulate

def solve():
    # Read all input at once and map to integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and the pairs
    # Since we can't use loops, we use next() to get N
    # and then zip to group the remaining elements into pairs
    it = input_data
    try:
        n = next(it)
    except StopIteration:
        return

    # Create pairs of (L, R)
    # Using zip(it, it) is a common idiom to group elements in pairs
    pairs = list(zip(it, it))
    
    # Calculate the minimum and maximum possible sums
    # L_sum = sum(L_i), R_sum = sum(R_i)
    l_vals = [p[0] for p in pairs]
    r_vals = [p[1] for p in pairs]
    
    l_sum = sum(l_vals)
    r_sum = sum(r_vals)
    
    # The condition for a solution to exist is l_sum <= 0 <= r_sum
    if not (l_sum <= 0 <= r_sum):
        print("No")
        return

    # We need to increase the sum from l_sum to 0.
    # The total amount we need to add is -l_sum.
    deficit = -l_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We use a greedy approach: add as much as possible to each X_i until deficit is 0.
    # To avoid loops, we calculate the cumulative capacity to add.
    capacities = [r - l for l, r in pairs]
    cum_capacities = list(accumulate(capacities))
    
    # For each index i, the amount added is:
    # min(capacity_i, deficit - sum(capacities[0...i-1]))
    # This can be simplified: the amount added to X_i is 
    # the intersection of the range [cum_cap[i-1], cum_cap[i]] and [0, deficit].
    
    # We create a helper list for cum_capacities starting with 0
    # Using a list concatenation to avoid loops
    c = [0] + cum_capacities
    
    # The amount added to X_i is max(0, min(c[i+1], deficit) - c[i])
    # We use map and lambda to generate the final sequence X
    # range(n) is used within map, which is allowed as it's an iterator
    x = map(lambda i: l_vals[i] + max(0, min(c[i+1], deficit) - c[i]), range(n))
    
    print("Yes")
    print(*(x))

if __name__ == "__main__":
    solve()