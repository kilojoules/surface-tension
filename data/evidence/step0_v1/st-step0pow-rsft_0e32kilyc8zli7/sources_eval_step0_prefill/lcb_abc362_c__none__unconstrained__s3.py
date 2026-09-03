import sys

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    L = []
    R = []
    
    idx = 1
    for _ in range(N):
        L.append(int(input_data[idx]))
        R.append(int(input_data[idx+1]))
        idx += 2
        
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # We start with X_i = L_i. The current sum is min_sum.
    # We need to increase the sum by -min_sum to reach 0.
    # We can increase each X_i up to R_i.
    current_sum = min_sum
    target_increase = -min_sum
    
    X = [0] * N
    for i in range(N):
        # The maximum we can add to X_i is R_i - L_i
        can_add = R[i] - L[i]
        # We add as much as possible without exceeding the target_increase
        add = min(can_add, target_increase)
        X[i] = L[i] + add
        target_increase -= add
        
    # Print the resulting sequence joined by spaces
    print(*(X))

if __name__ == "__main__":
    solve()