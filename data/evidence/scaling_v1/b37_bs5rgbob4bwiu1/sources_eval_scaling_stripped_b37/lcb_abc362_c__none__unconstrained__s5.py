import sys

def solve():
    # Read all input at once and split into a list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs (L_i, R_i) from the input list
    pairs = [
        (int(input_data[2*i + 1]), int(input_data[2*i + 2]))
        for i in range(N)
    ]
    
    # Calculate the minimum and maximum possible sums
    # min_sum = sum(L_i), max_sum = sum(R_i)
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start by setting every X_i to its minimum value L_i.
        # The current sum is min_sum. We need to increase this sum to 0.
        # The total amount we need to add is -min_sum.
        diff = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We greedily add as much as possible to each X_i until diff becomes 0.
        
        # Using a list comprehension to calculate the new X_i values.
        # Since we cannot use loops, we handle the state of 'diff' by 
        # recognizing that the amount added to X_i depends on the sum of 
        # (R_j - L_j) for j < i.
        
        # Let S_i be the prefix sum of (R_j - L_j).
        # The amount added to X_i is min(R_i - L_i, diff - sum_{j<i}(R_j - L_j))
        # However, a simpler way to implement this without a loop is to 
        # use the fact that we are distributing 'diff' across the available 
        # capacities (R_i - L_i).
        
        # We use a generator/map approach to maintain the state of remaining diff.
        # Since we need to track the remaining diff, we can use a mutable object 
        # (like a list or dictionary) to update the state inside a function.
        
        state = {'rem': diff}
        
        def allocate(p):
            l, r = p
            capacity = r - l
            add = min(capacity, state['rem'])
            state['rem'] -= add
            return l + add
        
        # Map the allocate function over all pairs
        result = map(allocate, pairs)
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()