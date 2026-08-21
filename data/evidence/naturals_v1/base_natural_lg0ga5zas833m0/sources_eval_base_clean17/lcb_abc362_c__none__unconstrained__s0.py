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
        
    # Calculate the minimum possible sum and maximum possible sum
    min_sum = sum(L)
    max_sum = sum(R)
    
    # The condition sum(X_i) = 0 can be satisfied if and only if 
    # the range [min_sum, max_sum] contains 0.
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # We start with the minimum possible values
    # Current sum is min_sum. We need to increase this sum to 0.
    current_sum = min_sum
    target_sum = 0
    diff = target_sum - current_sum
    
    X = [0] * N
    for i in range(N):
        # We can increase X[i] from L[i] up to R[i]
        # The maximum we can add to the sum using X[i] is (R[i] - L[i])
        can_add = R[i] - L[i]
        
        if diff <= 0:
            # No more increase needed
            X[i] = L[i]
        elif diff <= can_add:
            # This element can absorb the remaining difference
            X[i] = L[i] + diff
            diff = 0
        else:
            # Use the maximum capacity of this element and move to the next
            X[i] = R[i]
            diff -= can_add
            
    print(*(X))

if __name__ == "__main__":
    solve()