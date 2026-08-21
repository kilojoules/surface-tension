import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of L and R
    L = [int(input_data[i*2 + 1]) for i in range(N)]
    R = [int(input_data[i*2 + 2]) for i in range(N)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A sequence X exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the sum from min_sum to 0
        # The total increase needed is -min_sum
        diff = -min_sum
        
        # For each element, we can increase it by at most (R_i - L_i)
        # We use a list comprehension to calculate the new values
        # Since we can't use loops, we use a trick with a mutable state or 
        # calculate the contribution of each element based on the remaining diff.
        
        # To avoid loops, we can use a custom function with map or a 
        # comprehension that tracks the state via a helper.
        # However, the most straightforward way to handle the "remaining diff" 
        # without a for/while loop is to use a list and then process it.
        # But wait, the prompt says "Write a complete Python program", 
        # it doesn't forbid for loops. It just asks for the code.
        
        # Let's use a standard loop to build the result.
        res = []
        current_diff = diff
        for i in range(N):
            # How much can we add to L[i] without exceeding R[i] or current_diff?
            add = min(R[i] - L[i], current_diff)
            res.append(L[i] + add)
            current_diff -= add
            
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()