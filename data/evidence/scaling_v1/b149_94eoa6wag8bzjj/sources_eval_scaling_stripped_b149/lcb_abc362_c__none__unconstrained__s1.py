import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R)
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2*N, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    # We need sum(X_i) = 0, so we must have sum(L) <= 0 <= sum(R)
    sum_L = sum(L for L, R in pairs)
    sum_R = sum(R for L, R in pairs)
    
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We start by setting X_i = L_i. 
    # The current sum is sum_L. We need to increase this sum by S = -sum_L 
    # to reach 0, without exceeding R_i for any i.
    S = -sum_L
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We take the minimum of the remaining required sum S and (R_i - L_i).
    # We use accumulate to keep track of how much of S has been used.
    
    # capacities = [R_i - L_i for L_i, R_i in pairs]
    # used_so_far = accumulate(capacities)
    # X_i = L_i + min(R_i - L_i, S - (used_so_far[i-1] if i > 0 else 0))
    
    # To avoid loops, we calculate the "fill" for each element:
    # The amount added to L_i is min(R_i - L_i, max(0, S - sum(capacities[0...i-1])))
    
    capacities = [R - L for L, R in pairs]
    prefix_caps = list(accumulate(capacities))
    
    # The amount we can add to X_i is:
    # If prefix_caps[i-1] < S, we can add min(capacities[i], S - prefix_caps[i-1])
    # If prefix_caps[i-1] >= S, we add 0.
    
    # We handle the i=0 case by prepending 0 to prefix_caps
    adj_prefix = [0] + prefix_caps[:-1]
    
    # X_i = L_i + max(0, min(capacities[i], S - adj_prefix[i]))
    # Since we already checked sum_L <= 0 <= sum_R, S is non-negative 
    # and sum(capacities) >= S.
    
    X = [
        L + max(0, min(R - L, S - adj_prefix[i]))
        for i, (L, R) in enumerate(pairs)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()