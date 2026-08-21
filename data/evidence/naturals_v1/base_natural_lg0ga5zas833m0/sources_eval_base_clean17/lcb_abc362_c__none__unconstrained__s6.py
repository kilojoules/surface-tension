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
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A sequence X exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        print("Yes")
        
        # We start by setting X_i = L_i
        # Current sum is min_sum. We need to increase this sum to 0.
        # The amount we need to add is -min_sum.
        X = list(L)
        diff_needed = -min_sum
        
        for i in range(N):
            if diff_needed <= 0:
                break
            
            # The maximum we can add to X[i] is R[i] - L[i]
            can_add = R[i] - L[i]
            add = min(can_add, diff_needed)
            X[i] += add
            diff_needed -= add
            
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()