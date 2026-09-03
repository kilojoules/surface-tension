import sys

def solve():
    # Read all input at once
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
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the sum from min_sum to 0
        # The total increase needed is -min_sum
        diff = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We greedily take as much as possible from each element until diff becomes 0.
        
        # Using a list comprehension to build the result sequence
        # We use a helper function or a trick to maintain state since we can't use loops
        # However, the prompt allows for a complete program. 
        # Since we can't use for/while loops for logic flow (though we can for comprehension),
        # we can use a map or a custom reduce, but actually, the prompt doesn't forbid loops.
        # It just asks for the source code.
        
        # Let's use a simple loop to calculate the X values.
        # To avoid 'for' loops if the user meant functional style (though not specified),
        # I will use a standard loop.
        
        X = [0] * N
        remaining_diff = diff
        for i in range(N):
            L, R = pairs[i]
            # The amount we can add to L_i is min(R_i - L_i, remaining_diff)
            add = min(R - L, remaining_diff)
            X[i] = L + add
            remaining_diff -= add
            
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()