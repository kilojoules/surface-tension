import sys
from itertools import accumulate

def solve():
    # Read N and the pairs
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    pairs = [ (int(input_data[i]), int(input_data[i+1])) for i in range(1, 2*N, 2) ]
    
    # Calculate the minimum and maximum possible sums
    # L_sum = sum(L_i), R_sum = sum(R_i)
    L_vals = [p[0] for p in pairs]
    R_vals = [p[1] for p in pairs]
    
    sum_L = sum(L_vals)
    sum_R = sum(R_vals)
    
    # A solution exists if and only if 0 is within [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We need to increase the total sum from sum_L to 0.
        # The total increase needed is -sum_L.
        needed = -sum_L
        
        # For each i, the maximum we can increase X_i is (R_i - L_i).
        # We use a greedy approach: fill each X_i as much as possible.
        # To avoid loops, we calculate the cumulative capacity.
        capacities = [r - l for l, r in pairs]
        cum_cap = list(accumulate(capacities))
        
        # For each index i, the amount added is:
        # min(capacity[i], needed - sum(capacities[0...i-1]))
        # This is equivalent to:
        # max(0, min(capacity[i], needed - (cum_cap[i-1] if i>0 else 0)))
        
        # We can determine the value of X_i by looking at the cumulative capacity.
        # The total amount distributed up to index i is min(needed, cum_cap[i]).
        # The amount distributed specifically to index i is:
        # min(needed, cum_cap[i]) - (min(needed, cum_cap[i-1]) if i > 0 else 0)
        
        def get_X(i):
            current_cum = cum_cap[i]
            prev_cum = cum_cap[i-1] if i > 0 else 0
            added = min(needed, current_cum) - min(needed, prev_cum)
            return L_vals[i] + added

        X = list(map(get_X, range(N)))
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()