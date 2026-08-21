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
    # We need sum(X_i) = 0, so we must have sum(L_i) <= 0 <= sum(R_i)
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to increase the sum by target = 0 - min_sum.
    # For each i, we can increase X_i by at most (R_i - L_i).
    target = -min_sum
    
    # Calculate how much we can add to each L_i without exceeding R_i
    # and without exceeding the remaining target.
    # We use accumulate to track the total amount added so far.
    # The amount added to X_i is min(R_i - L_i, target - total_added_before)
    
    # Let f(acc, (L, R)) be the total added so far.
    # The increment for the current element is min(R - L, target - acc).
    def calc_total_added(acc, pair):
        L, R = pair
        return acc + min(R - L, target - acc)

    # accumulate returns the total added after each element.
    # We use a list comprehension to get the actual X_i values.
    totals = list(accumulate(pairs, calc_total_added, initial=0))
    
    # X_i = L_i + (total_added_after_i - total_added_before_i)
    # totals[i+1] is total added after element i, totals[i] is before.
    X = [
        pairs[i][0] + (totals[i+1] - totals[i]) 
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()