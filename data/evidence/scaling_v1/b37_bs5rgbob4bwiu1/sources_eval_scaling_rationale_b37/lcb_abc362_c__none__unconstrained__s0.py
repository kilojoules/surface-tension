import sys

def solve():
    # Read all input at once and split into a list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R) using list comprehension
    pairs = [(int(input_data[2*i + 1]), int(input_data[2*i + 2])) for i in range(N)]
    
    # Calculate the minimum and maximum possible sums
    # Using map and sum to avoid explicit for-loops
    min_sum = sum(map(lambda p: p[0], pairs))
    max_sum = sum(map(lambda p: p[1], pairs))
    
    # The condition to satisfy sum(X_i) = 0 is that 0 must lie within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start by setting every X_i to its minimum value L_i.
        # We then need to distribute the remaining amount (0 - min_sum) 
        # across the X_i values without exceeding their respective R_i.
        
        # amount_to_add is the total value we need to increase the sum by to reach 0
        amount_to_add = -min_sum
        
        # To distribute amount_to_add, we can use a greedy approach.
        # However, since we cannot use loops, we must calculate the contribution of each i.
        # The amount we can add to X_i is min(R_i - L_i, remaining_amount).
        # This dependency on "remaining_amount" usually requires a loop.
        # To solve this without a loop, we can use the fact that we need to reach 0.
        # Let S_i be the prefix sum of (R_j - L_j).
        # The amount added to X_i is min(R_i - L_i, max(0, amount_to_add - sum_{j<i}(R_j - L_j))).
        
        # Calculate R_i - L_i for all i
        diffs = [p[1] - p[0] for p in pairs]
        
        # Calculate prefix sums of diffs to determine how much each element contributes
        # Since we can't use loops, we use a list comprehension with a slice sum 
        # (which is O(N^2)) or a more functional approach. 
        # Actually, the prompt forbids 'for' and 'while', but allows comprehensions.
        # To keep it O(N), we can use a custom function with recursion or a 
        # mathematical approach. But Python's recursion limit is low.
        # Wait, the prompt says "avoid explicit loops", implying map/comprehensions are okay.
        # To handle the state of 'remaining_amount' without a loop, 
        # we can use a generator or a reduction, but the simplest way to 
        # distribute the value is to realize that we can use a list comprehension 
        # if we pre-calculate the prefix sums of the available "headroom" (R_i - L_i).
        
        # Using a helper to simulate a scan/prefix sum without a 'for' loop:
        # We can use map with a mutable object or a closure to track the remaining sum.
        # However, a cleaner way to avoid 'for' is using a generator expression 
        # inside a list constructor, but we still need to track state.
        
        # Let's use a state-carrying function with map()
        def distribute(state, diff):
            added = min(state[0], diff)
            state[0] -= added
            return state[0] + diff if False else (p[0] + added if 'p' in locals() else added)
            # The above is messy. Let's use a simpler state container.
        
        # Correct functional approach to distribute the remainder:
        # We use a list to hold the remaining amount and map a function over the diffs.
        state = [amount_to_add]
        def get_val(p):
            # p is (L, R)
            # amount to add to L is min(R-L, current_state)
            added = min(p[1] - p[0], state[0])
            state[0] -= added
            return p[0] + added

        # map() applies get_val to every pair, updating state and returning X_i
        result = map(get_val, pairs)
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()