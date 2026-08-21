import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [int(x) for x in input_data[1:]]
    L = pairs[0::2]
    R = pairs[1::2]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L)
    max_sum = sum(R)
    
    # The condition for a solution to exist is min_sum <= 0 <= max_sum
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with X_i = L_i. We need to add 'diff' to the total sum to reach 0.
    diff = -min_sum
    
    # We need to distribute 'diff' across the ranges (R_i - L_i).
    # To avoid loops, we calculate the prefix sum of the available capacities.
    # capacity_i = R_i - L_i
    capacities = [R[i] - L[i] for i in range(N)]
    
    # Using a list comprehension to calculate the amount to add to each L_i.
    # We need to know the sum of capacities before index i to determine 
    # if the remaining 'diff' is fully absorbed.
    # However, since we can't use loops or reduce, we can use a trick with 
    # a mutable object or a generator, but the prompt forbids loops.
    # A cleaner way is to use the fact that we can use map/zip/sum.
    
    # Let's use a helper function with a nonlocal variable inside a list comprehension
    # to simulate a stateful accumulator for the distribution of 'diff'.
    def distribute():
        state = {'rem': diff}
        def get_val(cap):
            add = min(state['rem'], cap)
            state['rem'] -= add
            return add
        return [get_val(c) for c in capacities]

    X_adjustments = distribute()
    X = [L[i] + X_adjustments[i] for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()