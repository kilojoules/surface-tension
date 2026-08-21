import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs (L_i, R_i)
    # input_data[1:] contains L1, R1, L2, R2, ...
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2 * N, 2)
    ]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values X_i = L_i
        # We need to increase the total sum from min_sum to 0.
        # The total increase needed is -min_sum.
        diff = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for index i is (R_i - L_i).
        # We greedily distribute the required 'diff' across the elements.
        
        # Using a list comprehension to calculate the new X_i values.
        # Since we cannot use loops, we use a helper function or a generator
        # to track the remaining diff. However, a simpler way is to 
        # use the fact that the increase for X_i is min(R_i - L_i, remaining_diff).
        # Because we need to track state (remaining_diff), we can use a 
        # custom function with a mutable accumulator or a generator.
        
        def distribute(pairs, total_diff):
            # We use a list to maintain state of remaining difference
            state = [total_diff]
            def calculate_x(p):
                L, R = p
                # Amount we can add to L to get closer to 0
                add = min(R - L, state[0])
                state[0] -= add
                return L + add
            
            return map(calculate_x, pairs)

        # Generate the sequence X
        X = distribute(pairs, diff)
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()