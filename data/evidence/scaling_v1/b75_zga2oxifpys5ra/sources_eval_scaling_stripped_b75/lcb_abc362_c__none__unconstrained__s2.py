import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of L and R
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2*N, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # The condition sum(X_i) = 0 is possible if and only if 
    # 0 is within the range [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We need to distribute the remaining sum needed to reach 0
    # Starting from the minimum possible sum:
    # Current sum = min_sum. We need to add S = -min_sum to reach 0.
    # For each X_i, we can increase it from L_i up to R_i.
    # The maximum we can add to X_i is (R_i - L_i).
    
    S = -min_sum
    # Calculate how much we can add to each L_i without exceeding S
    # available_increase_i = R_i - L_i
    # We use accumulate to track the total increase used so far
    increases = [R - L for L, R in pairs]
    acc_increases = list(accumulate(increases))
    
    # The amount added to X_i is:
    # min(available_increase_i, S - total_increase_before_i)
    # Which is equivalent to:
    # min(acc_increases[i], S) - max(acc_increases[i-1], 0) [with acc[-1]=0]
    
    # To avoid loops, we use a list comprehension to calculate X_i
    # X_i = L_i + (amount added)
    # The amount added to index i is:
    # clip(acc_increases[i], 0, S) - clip(acc_increases[i-1], 0, S)
    
    def clip(val):
        return max(0, min(val, S))
    
    # We handle the i-1 index by prepending 0 to the accumulated increases
    shifted_acc = [0] + acc_increases
    
    # X_i = L_i + clip(acc_i) - clip(acc_{i-1})
    result = [
        pairs[i][0] + clip(shifted_acc[i+1]) - clip(shifted_acc[i])
        for i in range(N)
    ]
    
    print("Yes")
    print(*(result))

if __name__ == "__main__":
    solve()