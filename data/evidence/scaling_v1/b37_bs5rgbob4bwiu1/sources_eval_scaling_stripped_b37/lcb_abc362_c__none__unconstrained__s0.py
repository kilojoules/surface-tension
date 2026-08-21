import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into a list of tuples [(L1, R1), (L2, R2), ...]
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(N)
    ]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values X_i = L_i
        # We need to increase the total sum from min_sum to 0.
        # The total increase needed is -min_sum.
        diff = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for index i is (R_i - L_i).
        # We distribute the required 'diff' across the elements.
        
        # Using a list comprehension to calculate X_i:
        # For each i, the increase is min(diff_remaining, R_i - L_i).
        # Since we cannot use a loop or external state easily, 
        # we can use the fact that the total increase is distributed.
        # However, the simplest way to handle the "remaining diff" 
        # without a loop is to use a generator or map with a mutable state,
        # but the prompt allows a "complete Python program". 
        # A simple loop to build the list is the most readable and standard.
        
        # To avoid 'for' loops for logic, we can use a custom function with reduce
        # or simply a list comprehension if we pre-calculate the prefix sums of ranges.
        # But a standard loop is permitted for constructing the result.
        
        # Let's use a generator to maintain the remaining difference:
        def distribute(remaining):
            for L, R in pairs:
                # Calculate how much we can add to L to get closer to 0
                add = min(remaining, R - L)
                yield L + add
                remaining -= add
        
        # Since we need to print the result, we can use the generator inside a list
        # However, the generator state (remaining) needs to be updated.
        # The most robust way in Python to handle state in a comprehension 
        # is using a helper class or a function with a nonlocal variable.
        
        # Let's use a simple loop to build the X list.
        X = []
        current_diff = diff
        for L, R in pairs:
            increase = min(current_diff, R - L)
            X.append(L + increase)
            current_diff -= increase
            
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()