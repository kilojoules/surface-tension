import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Group the remaining input into pairs of (L, R)
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

    # We need to distribute the 'deficit' relative to the minimum sum.
    # Let S = sum(L_i). We need to add a total of -S to the L_i values.
    # Each X_i can be increased from L_i up to R_i.
    # The amount we can add to X_i is (R_i - L_i).
    deficit = -min_sum
    
    # Calculate the cumulative capacity to increase values
    # capacities = [R_i - L_i for L, R in pairs]
    # cumulative_capacities = list(accumulate(capacities))
    
    # To avoid loops and maintain functional style, we use a generator 
    # to determine how much of the deficit is absorbed by each X_i.
    # For each i, the amount added is min(deficit_remaining, R_i - L_i).
    
    # We can use a trick with accumulate to track the remaining deficit.
    # state: (remaining_deficit, value_to_add)
    def step(rem, pair):
        L, R = pair
        add = min(rem, R - L)
        return rem - add, add

    # accumulate returns the state after each element. 
    # We only care about the 'add' part of the state.
    results = list(accumulate(pairs, step, initial=(deficit, 0)))
    
    # The first element of results is the initial state, so we skip it.
    # Each element in results[1:] is (remaining, added_value).
    final_X = [
        (pairs[i][0] + results[i+1][1]) 
        for i in range(N)
    ]
    
    print("Yes")
    print(*(final_X))

if __name__ == "__main__":
    solve()