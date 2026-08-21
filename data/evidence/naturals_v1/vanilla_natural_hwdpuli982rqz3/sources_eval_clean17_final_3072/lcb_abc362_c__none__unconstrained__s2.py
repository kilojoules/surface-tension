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
    sum_L = sum(p[0] for p in pairs)
    sum_R = sum(p[1] for p in pairs)
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with the minimum possible values
        # We need to increase the sum from sum_L to 0.
        # The total increase needed is -sum_L.
        diff = -sum_L
        
        # For each X_i, we can increase it from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We distribute 'diff' across the elements.
        
        # Using a list comprehension to calculate X_i without explicit loops
        # For each i, the increase is min(diff_remaining, R_i - L_i).
        # Since we can't easily track 'diff_remaining' in a comprehension, 
        # we use a trick with a helper function or a loop.
        # Given the constraints, a for loop is perfectly fine.
        
        X = [0] * N
        remaining = diff
        for i in range(N):
            L, R = pairs[i]
            # How much can we add to L_i without exceeding R_i or the remaining diff?
            add = min(remaining, R - L)
            X[i] = L + add
            remaining -= add
            
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()