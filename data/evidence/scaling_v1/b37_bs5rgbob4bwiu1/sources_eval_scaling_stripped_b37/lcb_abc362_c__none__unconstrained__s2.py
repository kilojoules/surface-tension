import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    # input_data[1:] contains L1, R1, L2, R2, ...
    # We use slice notation to get all Ls and all Rs
    L = [int(input_data[i]) for i in range(1, 2 * N, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 1, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A solution exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start by setting every X_i = L_i.
        # The current sum is sum_L. We need to increase this sum by S = 0 - sum_L.
        # We can increase each X_i up to R_i. The maximum increase for X_i is (R_i - L_i).
        S = -sum_L
        
        # Calculate the maximum possible increase for each element
        diffs = [r - l for l, r in zip(L, R)]
        
        # We need to distribute the total required increase S across the X_i.
        # For each i, we can add min(S_remaining, diff_i).
        # accumulate() with a custom function can track the remaining S.
        # However, a simpler way to think about it:
        # The amount added to X_i is min(diff_i, max(0, S - sum(diffs[i+1:])))
        # Actually, the simplest way is:
        # X_i = L_i + amount, where amount is capped by (R_i - L_i) 
        # and the total amount added across all i is exactly S.
        
        # Using accumulate to calculate the prefix sum of diffs:
        pref_diffs = accumulate(diffs)
        # The amount we can add to X_i is the difference between 
        # the total increase we can afford up to i and the total increase up to i-1.
        # Total increase allowed up to index i is min(S, prefix_sum_of_diffs[i])
        
        # We map the function f(prefix_sum) = min(S, prefix_sum) over the prefix sums
        # and then take the difference between adjacent terms.
        increases = map(lambda x: min(S, x), pref_diffs)
        
        # To get the individual increases:
        # diffs_actual[i] = min(S, pref[i]) - min(S, pref[i-1])
        # We can use zip to pair the shifted sequence.
        # We prepend 0 to the increases sequence to handle the first element.
        import itertools
        inc_vals = itertools.chain([0], increases)
        
        # Use a list comprehension to calculate X_i = L_i + (inc_i - inc_{i-1})
        # We use zip to iterate through L and the paired increases.
        # Since we cannot use loops, we use map/zip/comprehension.
        
        # Correct logic for X_i:
        # Let P_i be the prefix sum of (R_i - L_i).
        # The total increase added up to index i is min(S, P_i).
        # The increase added specifically to X_i is min(S, P_i) - min(S, P_{i-1}).
        
        # Let's redefine the increases logic without a loop:
        # 1. Calculate prefix sums of (R-L)
        # 2. Clip them at S
        # 3. Compute the difference between consecutive clipped sums
        
        # We use a list comprehension to build the final X sequence:
        # We need the clipped prefix sums as a list to access them by index
        clipped = [min(S, p) for p in pref_diffs]
        
        # X_i = L_i + (clipped[i] - (clipped[i-1] if i > 0 else 0))
        res = [L[i] + (clipped[i] - (clipped[i-1] if i > 0 else 0)) for i in range(N)]
        
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()