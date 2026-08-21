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
    for i in range(N):
        L.append(int(input_data[idx]))
        R.append(int(input_data[idx+1]))
        idx += 2
        
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # For a sequence X to exist such that sum(X) = 0,
    # 0 must lie within the range [min_sum, max_sum].
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # We start by setting X_i = L_i.
    # The current sum is min_sum. We need to increase this sum to 0.
    # The total amount we need to add is -min_sum.
    current_sum = min_sum
    needed = -current_sum
    
    X = [0] * N
    for i in range(N):
        # We can increase X_i from L_i up to R_i.
        # The maximum we can add to X_i is (R_i - L_i).
        can_add = R[i] - L[i]
        add = min(can_add, needed)
        
        X[i] = L[i] + add
        needed -= add
        
    # Print the resulting sequence
    print(*(X))

if __name__ == "__main__":
    solve()