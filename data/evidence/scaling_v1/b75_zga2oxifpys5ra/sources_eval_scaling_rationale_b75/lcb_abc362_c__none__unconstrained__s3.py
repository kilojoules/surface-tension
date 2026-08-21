import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    it = iter(input_data)
    
    # N is the first element
    N = next(it)
    
    # Use a list comprehension to pair L_i and R_i
    # Since we cannot use loops, we use a comprehension to consume the iterator
    pairs = [(L, R) for L, R in zip(it, it)]
    
    # Calculate the minimum and maximum possible sums
    # sum() is a built-in that replaces the need for a for-loop accumulator
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # Check if 0 is within the reachable range [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We need to distribute the remaining value needed to reach 0
    # Starting from the minimum possible sum:
    current_sum = min_sum
    target_diff = 0 - current_sum
    
    # We need to increase some X_i from L_i towards R_i until the sum is 0.
    # The amount we can increase X_i is (R_i - L_i).
    # We use a generator/list comprehension to determine the increase for each element.
    # However, the increase for element i depends on the remaining target_diff.
    # Since we cannot use loops or mutable state in a loop, we can't easily 
    # track 'remaining diff' across a comprehension.
    # But wait, we can use a trick with a mutable object (like a list) inside a 
    # comprehension or use a helper function with a closure/mutable state.
    
    state = {'rem': target_diff}
    
    def get_val(L, R):
        # Calculate how much we can add to L without exceeding R or the remaining diff
        add = min(R - L, state['rem'])
        state['rem'] -= add
        return L + add

    # Map the get_val function across all pairs
    # This effectively iterates through the list and updates the state
    result = [get_val(L, R) for L, R in pairs]
    
    print("Yes")
    print(*(result))

if __name__ == "__main__":
    solve()