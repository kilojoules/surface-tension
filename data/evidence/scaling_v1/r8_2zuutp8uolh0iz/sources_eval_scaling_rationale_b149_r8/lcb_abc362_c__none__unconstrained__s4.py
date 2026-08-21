import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs using slicing and map
    # Using list comprehensions to avoid explicit for loops
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 1, 2)]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L)
    max_sum = sum(R)
    
    # The condition for a solution to exist is that 0 must lie within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to increase the total sum from min_sum to 0.
        # The total amount we need to add is -min_sum.
        diff = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The amount we can add to X_i is (R_i - L_i).
        # However, we cannot add more than the remaining 'diff'.
        # Since we cannot use loops to maintain a running state of 'diff',
        # we calculate the prefix sums of the available capacities (R_i - L_i).
        
        capacities = [r - l for l, r in zip(L, R)]
        
        # Using a trick with a list comprehension and a helper to handle the 
        # distribution of 'diff' across the elements.
        # We need X_i = L_i + amount_added_i
        # amount_added_i = min(capacity_i, diff - sum(capacities[0...i-1]))
        
        # To avoid loops and recursion, we use a technique to track the 
        # cumulative capacity.
        import itertools
        prefix_capacities = list(itertools.accumulate(capacities))
        
        # The amount added to X_i is:
        # current_prefix_sum - previous_prefix_sum, but capped by the total diff.
        # Specifically: min(capacity_i, max(0, diff - previous_prefix_sum))
        
        # We can use a list comprehension to calculate X_i.
        # We handle the first element separately or use a 0-padded prefix list.
        prev_prefixes = [0] + prefix_capacities[:-1]
        
        X = [
            L[i] + min(capacities[i], max(0, diff - prev_prefixes[i]))
            for i in range(N)
        ]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()