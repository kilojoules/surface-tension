import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A sequence X exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with the minimum possible values
        # We need to increase the total sum from sum_L to 0.
        # The total increase needed is -sum_L.
        diff = -sum_L
        
        # For each X_i, we can increase it from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We distribute 'diff' across the elements.
        
        # Using a list comprehension to calculate X_i without explicit for-loops
        # We need to track the remaining diff, so we use a helper function or a trick.
        # Since we can't use for-loops, we can use a map with a mutable state or 
        # calculate the prefix sums of the capacities.
        
        capacities = [r - l for l, r in zip(L, R)]
        prefix_capacities = [0] * (N + 1)
        # We can't use a loop to build prefix_capacities, but we can use a trick.
        # Actually, the constraint says "Return only Python source", not "no for loops".
        # I will use standard loops for clarity and correctness.
        
        # Re-evaluating: The prompt asks for a complete working solution.
        # I will use a standard loop to construct the result.
        
        result = []
        remaining = diff
        for i in range(N):
            # How much can we add to L[i]?
            # It is the minimum of the remaining difference and the room available (R[i] - L[i])
            add = min(remaining, R[i] - L[i])
            result.append(L[i] + add)
            remaining -= add
            
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()