import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R)
    pairs = [
        (int(input_data[2 * i + 1]), int(input_data[2 * i + 2])) 
        for i in range(N)
    ]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the sum from min_sum to 0
        diff = 0 - min_sum
        
        # For each X_i, we can increase it from L_i up to R_i
        # The maximum increase for X_i is (R_i - L_i)
        # We distribute 'diff' across the elements
        
        # Using a list comprehension to calculate X_i without explicit for-loops
        # We need a way to track the remaining diff. 
        # Since we can't use for-loops, we can use a trick with a mutable object or 
        # simply calculate the prefix sums of the capacities.
        
        capacities = [p[1] - p[0] for p in pairs]
        # Compute prefix sums of capacities to find where the 'diff' is exhausted
        # However, the prompt allows "a complete Python program", and usually 
        # "no for/while loops" is a specific constraint for functional programming challenges.
        # Since no such constraint is mentioned here, I will use a standard loop.
        
        # Re-evaluating: The prompt does not forbid loops.
        
        X = [0] * N
        remaining = diff
        for i in range(N):
            L, R = pairs[i]
            # Increase L_i by the minimum of the remaining diff or the available capacity
            increase = min(remaining, R - L)
            X[i] = L + increase
            remaining -= increase
            
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()