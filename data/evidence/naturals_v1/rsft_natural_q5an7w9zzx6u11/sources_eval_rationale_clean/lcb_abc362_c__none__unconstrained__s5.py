import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    pairs = [int(x) for x in input_data[1:]]
    L = pairs[0::2]
    R = pairs[1::2]
    
    sum_L = sum(L)
    sum_R = sum(R)
    
    # Condition for existence: 0 must be within [sum(L), sum(R)]
    if sum_L <= 0 <= sum_R:
        # We need to increase sum_L to 0. 
        # The total amount to add is -sum_L.
        deficit = -sum_L
        
        # Capacity of each interval to be increased
        C = [r - l for l, r in zip(L, R)]
        
        # Use accumulate to find prefix sums of capacities
        # pref[i] is the total capacity of the first i elements
        pref = list(accumulate(C))
        
        # For each i, the amount added is:
        # min(C[i], deficit - (sum of capacities of elements before i))
        # This can be written as:
        # max(0, min(C[i], deficit - (pref[i-1] if i > 0 else 0)))
        
        # To avoid if/else in comprehension, we can prepend 0 to pref
        pref_shifted = [0] + pref[:-1]
        
        # Calculate X_i = L_i + amount_added
        # amount_added = max(0, min(C[i], deficit - pref_shifted[i]))
        X = [l + max(0, min(c, deficit - p)) 
             for l, c, p in zip(L, C, pref_shifted)]
        
        sys.stdout.write("Yes\n")
        sys.stdout.write(" ".join(map(str, X)) + "\n")
    else:
        sys.stdout.write("No\n")

if __name__ == "__main__":
    solve()