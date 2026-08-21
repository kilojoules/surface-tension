import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse L and R values into two separate lists using slicing and map
    L = list(map(int, input_data[1::2]))
    R = list(map(int, input_data[2::2]))
    
    # Calculate the minimum and maximum possible sums
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if not (sum_L <= 0 <= sum_R):
        print("No")
        return

    # We need to distribute the remaining value needed to reach 0
    # starting from the minimum possible sum (sum_L).
    # Target increase = 0 - sum_L = -sum_L
    target_increase = -sum_L
    
    # For each i, the maximum we can increase X_i from L_i is (R_i - L_i).
    # We use a greedy approach: increase X_i as much as possible until target_increase is 0.
    # Since we cannot use loops, we pre-calculate the cumulative capacity to increase.
    # However, a simpler functional way to distribute the remainder is to use 
    # the fact that X_i = L_i + min(R_i - L_i, remaining_target).
    # Because we can't maintain a 'remaining_target' variable in a loop,
    # we use the property that the total increase is distributed across the elements.
    
    # Let diff_i = R_i - L_i. We want to find x_i such that 0 <= x_i <= diff_i 
    # and sum(x_i) = target_increase.
    # We can achieve this by taking the prefix sums of diffs to determine where the 
    # target_increase is exhausted.
    
    diffs = [r - l for l, r in zip(L, R)]
    
    # To avoid loops, we can use a mathematical approach to distribute the target_increase:
    # X_i = L_i + (amount contributed to the target_increase)
    # The amount contributed by index i is:
    # min(diff_i, max(0, target_increase - sum(diffs[0...i-1])))
    
    # We can calculate prefix sums of diffs to find the cutoff point.
    # Using a list comprehension with a helper function or map to simulate the distribution:
    # Since we need the sum of previous elements, we can use a scan (prefix sum).
    # In Python, we can use itertools.accumulate for prefix sums.
    from itertools import accumulate
    
    prefix_diffs = list(accumulate(diffs))
    
    # X_i = L_i + contribution
    # contribution = min(diff_i, max(0, target_increase - prefix_diffs[i-1]))
    # For i=0, prefix_diffs[-1] is treated as 0.
    
    # We use a list comprehension to calculate X_i for all i.
    # We handle the i=0 case by using a list of prefix sums shifted by one.
    shifted_prefix = [0] + prefix_diffs[:-1]
    
    X = [
        l + min(d, max(0, target_increase - p))
        for l, d, p in zip(L, diffs, shifted_prefix)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()