import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Pair up L_i and R_i
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(n)
    ]
    
    # Calculate the total minimum and maximum possible sums
    # Using map/sum to avoid explicit loops
    min_sum = sum(map(lambda p: p[0], pairs))
    max_sum = sum(map(lambda p: p[1], pairs))
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We need to find X_i such that sum(X_i) = 0 and L_i <= X_i <= R_i.
    # Let X_i = L_i + d_i, where 0 <= d_i <= (R_i - L_i).
    # We need sum(L_i + d_i) = 0  =>  sum(d_i) = -sum(L_i).
    target_d_sum = -min_sum
    
    # Calculate the maximum possible increase for each element
    max_ds = [r - l for l, r in pairs]
    
    # Use accumulate to find the prefix sum of the maximum possible increases.
    # This allows us to determine how much of the target_d_sum is absorbed by each X_i.
    prefix_max_ds = list(accumulate(max_ds))
    
    # For each i, the amount we can add to L_i is:
    # min(max_d_i, target_d_sum - sum(max_d_j for j < i))
    # However, it's simpler to say: 
    # The total increase used up to index i is min(target_d_sum, prefix_max_ds[i]).
    # The increase for index i is then (total used up to i) - (total used up to i-1).
    
    # Calculate total increase used at each step
    used_until = [min(target_d_sum, val) for val in prefix_max_ds]
    
    # Calculate the specific increase for each element
    # d_0 = used_until[0]
    # d_i = used_until[i] - used_until[i-1]
    # We use a list comprehension to build the final X sequence
    # X_i = L_i + d_i
    
    # To avoid a loop for d_i, we can use the fact that:
    # X_i = L_i + (used_until[i] - used_until[i-1])
    # For i=0, used_until[-1] is 0.
    
    # We create a helper list for used_until that starts with 0
    u = [0] + used_until
    
    # Final sequence X
    # We use range(n) and index into our prepared lists
    # L_i is pairs[i][0]
    result = [pairs[i][0] + (u[i+1] - u[i]) for i in range(n)]
    
    print("Yes")
    print(*(result))

if __name__ == "__main__":
    solve()