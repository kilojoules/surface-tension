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
        R.append(int(input_data[idx+1]))
        idx += 2
        
    # The sum of X_i must be 0.
    # The minimum possible sum is sum(L_i) and maximum is sum(R_i).
    # A solution exists if and only if sum(L_i) <= 0 <= sum(R_i).
    
    min_sum = sum(L)
    max_sum = sum(R)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # Start by setting each X_i to its lower bound L_i.
    # We need to increase the sum from min_sum to 0.
    X = list(L)
    current_sum = min_sum
    target_sum = 0
    needed = target_sum - current_sum
    
    for i in range(N):
        if needed <= 0:
            break
        
        # The amount we can increase X_i is R_i - L_i
        can_add = R[i] - L[i]
        add = min(can_add, needed)
        X[i] += add
        needed -= add
        
    # Print the resulting sequence
    print(*(X))

if __name__ == "__main__":
    solve()