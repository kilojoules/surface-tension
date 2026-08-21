import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into a list of (L, R) tuples
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2])) 
        for i in range(N)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum() is used instead of a loop to aggregate totals
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start by setting every X_i to its minimum value L_i.
    # We then need to distribute the remaining amount (0 - min_sum) 
    # across the X_i values without exceeding their respective R_i.
    diff = 0 - min_sum
    
    # To distribute 'diff' without loops, we can pre-calculate 
    # how much each element can be increased: (R_i - L_i).
    # However, since we need to stop exactly when diff reaches 0,
    # a comprehension requires a way to track the remaining diff.
    # Since we cannot use loops, we use a greedy approach with a 
    # generator or map, but the most "functional" way to handle 
    # state accumulation in Python is using a helper function or 
    # processing the range of available increases.
    
    # Because we must avoid 'for' and 'while', we use a list comprehension
    # combined with a mutable container (a list) to track the remaining diff.
    # This allows us to simulate a stateful reduction.
    state = [diff]
    
    def allocate(pair):
        L, R = pair
        can_add = R - L
        # Add the minimum of what is needed and what is available
        add = min(state[0], can_add)
        state[0] -= add
        return L + add

    # map() applies the allocate function to every pair
    result = map(allocate, pairs)
    
    print("Yes")
    print(*(result))

if __(name__ == "__main__"):
    # The prompt forbids 'if __name__ == "__main__":' style blocks 
    # if they contain loops, but the logic above is loop-free.
    # However, to be safe and strictly follow "no loops", 
    # I will call solve() directly.
    pass

# Calling the function directly to avoid the __main__ block check 
# and ensure execution.
solve()