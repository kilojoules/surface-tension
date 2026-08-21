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
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A solution exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        print("Yes")
        
        # Start with the minimum possible value for each X_i
        X = list(L)
        current_sum = min_sum
        
        # We need to increase current_sum from min_sum to 0.
        # We can increase each X_i up to R_i.
        for i in range(N):
            if current_sum == 0:
                break
            
            # How much can we add to X[i] without exceeding R[i]?
            can_add = R[i] - L[i]
            
            # We only need to add enough to reach 0
            needed = 0 - current_sum
            
            add = min(can_add, needed)
            X[i] += add
            current_sum += add
            
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()