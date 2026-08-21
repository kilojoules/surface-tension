import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    # Using list comprehension instead of for-loops
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i + 1]) for i in range(1, 2 * N + 1, 2)]
    
    min_sum = sum(L)
    max_sum = sum(R)
    
    # Check if 0 is reachable within the bounds of the sums
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values.
        # We need to add 'diff' to the elements to reach a total sum of 0.
        diff = -min_sum
        
        # To avoid loops, we use a state-carrying mechanism.
        # Since we need to track the remaining diff across elements, 
        # and map/comprehensions are stateless, we can use a mutable object (list)
        # to track the remaining difference to be distributed.
        state = [diff]
        
        def allocate(l_val, r_val):
            # Calculate how much we can add to the current L_i without exceeding R_i
            # and without exceeding the remaining difference.
            can_add = min(r_val - l_val, state[0])
            state[0] -= can_add
            return l_val + can_add

        # Use map to apply the allocation logic to all pairs
        result = map(allocate, L, R)
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()