import sys

def solve():
    # Read N from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    L = []
    R = []
    
    idx = 1
    for _ in range(N):
        L.append(int(input_data[idx]))
        R.append(int(input_data[idx + 1]))
        idx += 2
        
    # The minimum possible sum is the sum of all lower bounds L_i
    # The maximum possible sum is the sum of all upper bounds R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # For a sum of 0 to be achievable, 0 must be within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # We start by setting each X_i to its lower bound L_i.
    # The current sum is min_sum. We need to increase this sum to 0.
    # We can increase each X_i up to R_i.
    X = list(L)
    current_sum = min_sum
    target_sum = 0
    diff_needed = target_sum - current_sum
    
    # Greedy adjustment: increase X_i as much as possible until diff_needed is 0
    for i in range(N):
        if diff_needed <= 0:
            break
        
        # Maximum we can add to X[i] is R[i] - L[i]
        can_add = R[i] - L[i]
        add = min(can_add, diff_needed)
        
        X[i] += add
        diff_needed -= add
        
    # Output the result
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()