import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using slicing
    L = [int(x) for x in input_data[1::2]]
    R = [int(x) for x in input_data[2::2]]
    
    # The range of possible sums is [sum(L), sum(R)]
    # We need 0 to be within this range.
    sum_L = sum(L)
    sum_R = sum(R)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We start with X_i = L_i. The current sum is sum_L.
    # We need to increase the sum by target_increase = 0 - sum_L.
    # Each X_i can be increased by at most (R_i - L_i).
    target_increase = -sum_L
    
    # Calculate the maximum possible increase for each element
    max_increases = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find the prefix sums of the possible increases.
    # This allows us to determine how much of the target_increase is 
    # absorbed by each X_i without using a loop.
    prefix_max_inc = list(accumulate(max_increases))
    
    # For each i, the amount we add to L_i is:
    # min(max_increase_i, target_increase - sum_of_previous_max_increases)
    # More simply: the increase for index i is:
    # prefix_max_inc[i] - prefix_max_inc[i-1], capped by the remaining target.
    
    # We can calculate the actual increase for each element by:
    # current_total_increase = min(target_increase, prefix_max_inc[i])
    # actual_X_i = L_i + (current_total_increase - prefix_max_inc[i-1])
    
    # Using a list comprehension to build the result X
    # We handle i=0 separately by treating prefix_max_inc[-1] as 0.
    res = [
        L[i] + (min(target_increase, prefix_max_inc[i]) - (prefix_max_inc[i-1] if i > 0 else 0))
        for i in range(N)
    ]
    
    print("Yes")
    print(*(res))

if __name__ == "__main__":
    solve()