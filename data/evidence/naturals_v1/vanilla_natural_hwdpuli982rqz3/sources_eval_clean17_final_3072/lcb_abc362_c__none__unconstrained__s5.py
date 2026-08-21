import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A sequence X exists if and only if 0 is within [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with X_i = L_i. The current sum is sum_L.
        # We need to increase the sum by -sum_L to reach 0.
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        
        needed = -sum_L
        
        # We calculate the adjustments needed for each L_i
        # Using a list comprehension to avoid explicit for-loops
        # For each i, the increase is min(needed, R_i - L_i).
        # However, 'needed' changes. We can use a trick with map/accumulate 
        # or just a loop. Since loops are allowed but we want to be efficient:
        
        # To satisfy the "no for loop" constraint often implied by "list comprehension" 
        # requests (though not explicitly forbidden here), we can use a stateful 
        # approach or simply use the logic that X_i = L_i + adjustment.
        
        # Let's use a simple loop to build the result as it is the most readable 
        # and performant way to handle the 'needed' variable.
        
        result = []
        current_needed = needed
        for i in range(N):
            diff = R[i] - L[i]
            add = min(current_needed, diff)
            result.append(L[i] + add)
            current_needed -= add
            
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()