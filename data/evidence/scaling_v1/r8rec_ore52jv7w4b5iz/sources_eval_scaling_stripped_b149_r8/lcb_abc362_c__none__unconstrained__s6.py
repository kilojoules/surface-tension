import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using slicing
    L = [int(x) for x in input_data[1::2]]
    R = [int(x) for x in input_data[2::2]]
    
    # The range of possible sums is [sum(L), sum(R)]
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with the minimum possible values X_i = L_i.
        # The current sum is sum_L. We need to increase this sum to 0.
        # The total amount we need to add is -sum_L.
        needed = -sum_L
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We use a list comprehension to calculate the actual X_i.
        # However, since we cannot use loops, we must track the 'needed' 
        # amount across the sequence. This is tricky without loops.
        # But we can use the property that we fill the gaps greedily.
        
        # Let S_i be the prefix sum of (R_i - L_i).
        # The amount added to X_i is min(R_i - L_i, max(0, needed - prefix_sum_{i-1}))
        # Instead, we can use the fact that we need to distribute 'needed' 
        # across the capacities (R_i - L_i).
        
        # We can use a generator with a mutable state (like a list) inside 
        # a comprehension, but that's essentially a loop. 
        # A cleaner way is to use the fact that we can use a helper function 
        # with a closure or a class to maintain state, but the constraint 
        # says "no loops". 
        # Actually, we can use a list to store the 'remaining' needed amount 
        # and update it using a side-effect in a list comprehension.
        
        state = [needed]
        def get_x(i):
            capacity = R[i] - L[i]
            add = min(capacity, state[0])
            state[0] -= add
            return L[i] + add
        
        # Map the helper function across the range of N
        X = list(map(get_x, range(N)))
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()