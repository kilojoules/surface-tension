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
    
    # To satisfy sum(X_i) = 0, we must have:
    # sum(L_i) <= 0 <= sum(R_i)
    # Because the smallest possible sum is sum(L_i) and largest is sum(R_i)
    
    min_sum = sum(L)
    max_sum = sum(R)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # We start with X_i = L_i. 
    # The current sum is min_sum. We need to increase this sum to 0.
    # The amount we need to add is -min_sum.
    # We can increase each X_i up to R_i.
    
    X = list(L)
    needed = -min_sum
    
    for i in range(N):
        if needed <= 0:
            break
        
        # The maximum we can add to X[i] is R[i] - L[i]
        can_add = R[i] - L[i]
        add = min(can_add, needed)
        X[i] += add
        needed -= add
        
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()