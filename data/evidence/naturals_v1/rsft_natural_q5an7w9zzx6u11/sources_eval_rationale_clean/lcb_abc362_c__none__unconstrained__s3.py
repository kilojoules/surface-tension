import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into two lists: L and R
    # Using list comprehensions to avoid explicit loops
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L)
    max_sum = sum(R)
    
    # The condition for a solution to exist is that 0 must be within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We start with X_i = L_i. We need to add 'surplus' to reach a total sum of 0.
    # Total needed increase = 0 - min_sum
    surplus = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i)
    capacities = [r - l for l, r in zip(L, R)]
    
    # We use accumulate to find how much of the surplus is consumed by each element.
    # cumulative_capacities[i] is the total capacity of elements 0 to i.
    cumulative_capacities = list(accumulate(capacities))
    
    # For each element i, the amount added is:
    # min(capacity_of_i, remaining_surplus)
    # This can be calculated as:
    # amount_added_up_to_i = min(surplus, cumulative_capacities[i])
    # amount_added_to_i = amount_added_up_to_i - amount_added_up_to_i_minus_1
    
    # Calculate the total amount allocated to each index
    allocated_totals = [min(surplus, c) for c in cumulative_capacities]
    
    # To get the individual X_i, we subtract the previous allocated total from the current
    # and add it to L_i.
    # X_i = L_i + (allocated_totals[i] - allocated_totals[i-1])
    # For i=0, allocated_totals[-1] is 0.
    
    # Using a list comprehension to build the final sequence X
    # We prepend 0 to allocated_totals to handle the i=0 case easily
    totals = [0] + allocated_totals
    X = [L[i] + (totals[i+1] - totals[i]) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()