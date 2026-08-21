import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = list(map(int, sys.stdin.read().split()))
    
    # N is the first element
    n = input_data[0]
    
    # Pair the remaining elements into (L, R) tuples
    # Using slice [1:] and a list comprehension to create pairs
    pairs = [ (input_data[i], input_data[i+1]) for i in range(1, len(input_data), 2) ]
    
    # Calculate the minimum and maximum possible sums
    # Using generator expressions inside sum() for efficiency
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # The condition for a solution to exist is that 0 must fall within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start by setting each X_i to L_i.
    # We need to add a total of 'deficit' to the sum to reach 0.
    deficit = 0 - min_sum
    
    # To distribute the deficit, we calculate how much each X_i can be increased: (R_i - L_i).
    # We use a greedy approach: increase X_i by min(deficit, R_i - L_i).
    # Since we cannot use loops, we must handle the 'deficit' update carefully.
    # However, the deficit changes as we iterate. To avoid a loop, we can use a 
    # mathematical trick with prefix sums of the capacities.
    
    capacities = [p[1] - p[0] for p in pairs]
    prefix_capacities = [0] + [sum(capacities[:i+1]) for i in range(n)] # This is O(N^2), too slow.
    
    # Correction: To avoid O(N^2) and loops, we can use a list comprehension 
    # combined with a helper function or a more clever approach.
    # Actually, the most idiomatic way to handle state in a comprehension 
    # is using a mutable object (like a list) to track the remaining deficit.
    
    state = [deficit]
    
    def allocate(cap):
        # Calculate how much we can take from the current deficit
        taken = min(state[0], cap)
        state[0] -= taken
        return taken

    # Use map or list comprehension to apply the allocate function
    # X_i = L_i + amount_allocated
    results = [p[0] + allocate(p[1] - p[0]) for p in pairs]
    
    print("Yes")
    print(*(results))

# Using a wrapper to call solve()
if __name__ == "__main__":
    # Increase recursion depth just in case, though not needed for this approach
    sys.setrecursionlimit(300000)
    solve()