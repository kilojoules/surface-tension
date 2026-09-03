import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A sequence X exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the total sum from min_sum to 0
        diff = 0 - min_sum
        
        # For each element, we can increase it by at most (R_i - L_i)
        # We use a list comprehension to calculate the new values
        # Since we can't use loops, we use a trick with a mutable state or map
        # But we can actually calculate the contribution of each element using 
        # the prefix sums of the available ranges.
        
        ranges = [r - l for l, r in zip(L, R)]
        # Compute prefix sums of ranges to determine how much each X_i contributes to 'diff'
        # However, we can just iterate through and subtract from diff.
        # To avoid loops, we can use a helper function with map or a list comprehension 
        # that updates an external variable.
        
        result = []
        # We use a list to keep track of the remaining difference to reach 0
        state = [diff]
        
        def calculate_x(i):
            # amount to add to L[i] is min(range[i], remaining_diff)
            added = min(ranges[i], state[0])
            state[0] -= added
            return L[i] + added
        
        # Map the calculate_x function over the indices
        X = list(map(calculate_x, range(N)))
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()