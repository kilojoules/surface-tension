import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [int(x) for x in input_data[1:]]
    L = pairs[0::2]
    R = pairs[1::2]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start by setting X_i = L_i. 
    # We need to increase the total sum from min_sum to 0.
    # The total amount we need to add is -min_sum.
    needed = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We can use a greedy approach: add as much as possible to each X_i 
    # until 'needed' becomes 0.
    
    # To avoid loops, we calculate the prefix sums of the available capacities (R_i - L_i).
    # However, a simpler loop-free way to distribute 'needed' is to realize that
    # X_i = L_i + min(R_i - L_i, remaining_needed).
    # Since we can't use loops, we can use a technique with a running total.
    # But wait, the simplest loop-free way to implement this logic is to 
    # realize that we can just cap the total increase at 'needed'.
    
    # Let S_i be the prefix sum of (R_i - L_i).
    # The amount added to X_i is:
    # min(R_i - L_i, max(0, needed - S_{i-1}))
    
    capacities = [r - l for l, r in zip(L, R)]
    
    # Using a list comprehension to calculate prefix sums is tricky without loops.
    # But we can use a trick with a mutable object or just use the fact that 
    # we can distribute 'needed' by taking the minimum of the capacity and the 
    # remaining needed amount.
    
    # Since we must avoid loops, we can use a generator with a state-carrying function
    # or use the fact that we can use a list and a helper function.
    # Actually, the most idiomatic "no-loop" way to handle state in Python 
    # for this specific problem is to use a closure and map().
    
    def distribute(cap):
        # This closure tracks the remaining 'needed' amount
        state = {'rem': needed}
        def calc(c):
            add = min(c, state['rem'])
            state['rem'] -= add
            return add
        return calc

    # Create the distribution function
    dist_func = distribute(None)
    # Use map to apply the function to all capacities
    added_amounts = list(map(dist_func, capacities))
    
    # Final X_i = L_i + added_amount_i
    X = [l + a for l, a in zip(L, added_amounts)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()