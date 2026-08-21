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
        
    # The range of possible sums for X is [sum(L), sum(R)]
    min_sum = sum(L)
    max_sum = sum(R)
    
    # For a solution to exist, 0 must lie within the range [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # We start by setting each X_i to its minimum possible value L_i.
    # Current sum is min_sum. We need to increase this sum to 0.
    current_sum = min_sum
    target_increase = -min_sum # How much we need to add to get to 0
    
    X = [0] * N
    for i in range(N):
        # For each element, we can increase it from L_i up to R_i.
        # The maximum possible increase for X_i is (R_i - L_i).
        possible_increase = R[i] - L[i]
        
        # We add as much as possible without exceeding the target_increase.
        add = min(possible_increase, target_increase)
        X[i] = L[i] + add
        target_increase -= add
        
    # Print the sequence X separated by spaces
    print(*(X))

if __name__ == "__main__":
    solve()