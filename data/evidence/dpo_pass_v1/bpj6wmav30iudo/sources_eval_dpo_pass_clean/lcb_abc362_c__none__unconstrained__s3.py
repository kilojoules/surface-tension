import sys
from functools import reduce

def solve():
    # Read all input at once and map to integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use an iterator to consume the input stream
    it = iter(input_data)
    
    # Extract N and the pairs (L, R)
    # Since we cannot use loops, we use a list comprehension to pair L and R
    # We use next(it) to get N first
    # Note: Since we need N to build the list, we handle it carefully
    # We'll read the rest of the data into a list first
    data = list(it)
    N = data[0]
    pairs = [(data[i], data[i+1]) for i in range(1, 2*N, 2)]
    
    # Calculate the minimum and maximum possible sums
    # reduce is used to sum the Ls and Rs
    min_sum = reduce(lambda a, b: a + b[0], pairs, 0)
    max_sum = reduce(lambda a, b: a + b[1], pairs, 0)
    
    # Check if 0 is within the reachable range [min_sum, max_sum]
    # If not, print No. Otherwise, calculate the required adjustment.
    # The adjustment is how much we need to add to min_sum to reach 0.
    # adjustment = 0 - min_sum
    adjustment = -min_sum
    
    # To distribute the adjustment:
    # For each X_i, we start with L_i. 
    # We can add at most (R_i - L_i) to each X_i.
    # Since we can't use loops to track the remaining adjustment,
    # we use a trick: we calculate the prefix sums of the capacities (R_i - L_i).
    # However, a simpler way is to realize that we need to fill the capacities
    # one by one. 
    # Let C_i = R_i - L_i. We need sum(delta_i) = adjustment, where 0 <= delta_i <= C_i.
    
    # We can use a list comprehension with a helper function or a closure 
    # to track the remaining adjustment. 
    # Since we cannot use loops, we use a mutable object (a list) to track state 
    # inside a list comprehension.
    
    state = [adjustment]
    
    # Helper to calculate delta for each element
    # delta = min(capacity, remaining_adjustment)
    # remaining_adjustment -= delta
    get_delta = lambda capacity: (
        (state.__setitem__(0, state[0] - min(capacity, max(0, state[0]))), 
         min(capacity, max(0, state[0])))[-1]
    )
    
    # Construct X_i = L_i + delta_i
    # We use a list comprehension to trigger the state update
    X = [L + get_delta(R - L) for L, R in pairs]
    
    # Final output logic using a conditional expression
    sys.stdout.write(
        "No\n" if not (min_sum <= 0 <= max_sum) 
        else "Yes\n" + " ".join(map(str, X)) + "\n"
    )

if __name__ == "__main__":
    solve()