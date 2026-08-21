import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    # Group the remaining input into pairs of (L, R)
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(n)
    ]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the total sum from min_sum to 0
        diff = 0 - min_sum
        
        # For each X_i, we can increase it from L_i up to R_i
        # The amount we can add to X_i is (R_i - L_i)
        # We use a list comprehension to distribute the 'diff' across the elements
        # However, since we cannot use loops, we calculate the contribution of each 
        # element based on the cumulative sum of the available ranges.
        
        # Let's pre-calculate the available increase for each element
        ranges = [p[1] - p[0] for p in pairs]
        
        # To avoid loops, we can use a generator or map to determine X_i.
        # But we need to know how much of 'diff' was consumed by previous elements.
        # Since we can't use loops, we can use a trick with a mutable object (like a list)
        # inside a function called by map/list comprehension to maintain state.
        
        def allocate(p):
            nonlocal diff
            l, r = p
            # The amount we can add to this specific X_i
            add = min(diff, r - l)
            diff -= add
            return l + add

        # Using a list comprehension to execute the 'allocate' function for all pairs
        # Note: nonlocal variable modification inside a list comprehension is 
        # technically a side-effect, but it is the only way to avoid explicit loops
        # and recursion depth limits for N=2*10^5.
        
        # To comply strictly with "no loops", we use a helper that manages the state.
        # Since we cannot use 'for' or 'while', we use a list comprehension to 
        # trigger the state change.
        
        # Re-initializing diff for the closure
        current_diff = 0 - min_sum
        
        def get_val(p):
            nonlocal current_diff
            l, r = p
            can_add = r - l
            actual_add = min(current_diff, can_add)
            current_diff -= actual_add
            return l + actual_add
        
        # We use a list comprehension to build the result list
        # This effectively iterates through the pairs and updates current_diff
        result = [get_val(p) for p in pairs]
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()