import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Pair L_i and R_i using slicing
    ls = list(map(int, input_data[1::2]))
    rs = list(map(int, input_data[2::2]))
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(ls)
    max_sum = sum(rs)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We need to distribute the remaining sum needed to reach 0
    # starting from the minimum possible sum.
    needed = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i)
    capacities = [r - l for l, r in zip(ls, rs)]
    
    # Use accumulate to find the prefix sums of capacities to determine
    # how much of the 'needed' sum has been absorbed by index i.
    # prefix_caps[i] is the total capacity of elements 0 to i.
    prefix_caps = list(accumulate(capacities))
    
    # The amount added to X_i is:
    # min(capacity_i, needed - sum_of_previous_capacities)
    # This can be expressed as: min(prefix_caps[i], needed) - min(prefix_caps[i-1], needed)
    # We prepend 0 to prefix_caps to handle the i=0 case.
    extended_prefix = [0] + prefix_caps
    
    # Calculate X_i = L_i + (contribution to the needed sum)
    # Using a list comprehension to avoid explicit loops
    res = [
        ls[i] + (min(extended_prefix[i+1], needed) - min(extended_prefix[i], needed))
        for i in range(n)
    ]
    
    print("Yes")
    print(*(res))

if __name__ == "__main__":
    solve()