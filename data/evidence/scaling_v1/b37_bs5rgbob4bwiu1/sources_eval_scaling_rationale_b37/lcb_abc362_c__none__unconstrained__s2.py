import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    # Using slice and zip to avoid loops
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    sum_L = sum(L)
    sum_R = sum(R)
    
    # Condition for existence: 0 must be within [sum(L), sum(R)]
    if sum_L <= 0 <= sum_R:
        # We need to distribute the difference (0 - sum_L) across the elements.
        # Each X_i = L_i + diff_i, where 0 <= diff_i <= R_i - L_i.
        # Total diff needed is -sum_L.
        target_diff = -sum_L
        
        # Calculate the maximum possible increase for each element
        max_diffs = [r - l for l, r in zip(L, R)]
        
        # Use accumulate to find how much of the target_diff is consumed up to index i
        # accumulate([a, b, c]) -> a, a+b, a+b+c
        acc_max_diffs = accumulate(max_diffs)
        
        # For each i, the amount we can add to L_i is:
        # min(max_diff_i, target_diff - sum_of_previous_max_diffs)
        # However, a simpler way: 
        # X_i = L_i + (amount of target_diff remaining to be distributed)
        # Let S_i be the prefix sum of max_diffs.
        # The amount added to X_i is min(max_diff_i, max(0, target_diff - S_{i-1}))
        
        # To implement this without loops, we use the property:
        # The total amount added to the first k elements is min(S_k, target_diff).
        # Therefore, X_i = L_i + (min(S_i, target_diff) - min(S_{i-1}, target_diff))
        
        # Create a sequence of S_i starting with 0
        S = (0,) + tuple(acc_max_diffs)
        
        # Calculate X_i using a list comprehension
        # X_i = L_i + min(S[i+1], target_diff) - min(S[i], target_diff)
        X = [L[i] + min(S[i+1], target_diff) - min(S[i], target_diff) for i in range(N)]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()