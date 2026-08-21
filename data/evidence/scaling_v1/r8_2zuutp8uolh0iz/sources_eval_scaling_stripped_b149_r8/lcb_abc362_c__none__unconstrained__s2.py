import sys
from itertools import accumulate

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using slicing
    L = [int(x) for x in input_data[1::2]]
    R = [int(x) for x in input_data[2::2]]
    
    # Calculate the minimum and maximum possible sums
    # Using sum() is efficient and allowed
    sum_L = sum(L)
    sum_R = sum(R)
    
    # The condition for a solution to exist is sum(L) <= 0 <= sum(R)
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We need to find X_i such that L_i <= X_i <= R_i and sum(X_i) = 0.
    # Start with X_i = L_i. The current sum is sum_L.
    # We need to add S = -sum_L to the elements to reach a total sum of 0.
    # Each X_i can be increased by at most (R_i - L_i).
    S = -sum_L
    
    # Calculate the available "headroom" for each interval
    diffs = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find the prefix sums of the headrooms.
    # prefix_sums[i] is the total amount we could have added up to index i.
    prefix_sums = list(accumulate(diffs))
    
    # We need to distribute S across the X_i.
    # For the i-th element, the amount added is:
    # min(diffs[i], S - (sum of diffs of previous elements))
    # This is equivalent to: min(prefix_sums[i], S) - min(prefix_sums[i-1], S)
    
    # To avoid a loop, we create a shifted version of prefix_sums for the i-1 term.
    # We prepend 0 to prefix_sums to handle the i=0 case.
    ps = [0] + prefix_sums
    
    # The amount added to X_i is min(ps[i+1], S) - min(ps[i], S)
    # We use a list comprehension to calculate the final X values.
    X = [L[i] + (min(ps[i+1], S) - min(ps[i], S)) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()