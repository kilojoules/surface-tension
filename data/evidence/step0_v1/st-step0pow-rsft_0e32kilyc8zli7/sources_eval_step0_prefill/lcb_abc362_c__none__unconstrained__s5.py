import sys

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    L = []
    R = []
    
    # Parse the L_i and R_i pairs
    for i in range(N):
        L.append(int(input_data[2*i + 1]))
        R.append(int(input_data[2*i + 2]))
    
    # The minimum possible sum is the sum of all lower bounds
    # The maximum possible sum is the sum of all upper bounds
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # Start with the minimum possible values
        X = list(L)
        current_sum = min_sum
        
        # We need to increase the sum from min_sum to 0
        # We can increase each X_i up to R_i
        needed = -current_sum
        
        for i in range(N):
            if needed <= 0:
                break
            
            # The maximum we can add to X_i is (R_i - L_i)
            can_add = R[i] - L[i]
            add = min(can_add, needed)
            
            X[i] += add
            needed -= add
            
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()