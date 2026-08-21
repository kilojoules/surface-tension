import sys
from itertools import accumulate

def solve():
    # Read all input data
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
        # We need to add 'rem' to the sum of Ls to reach 0
        rem = -sum_L
        
        # For each i, the amount we can add is R[i] - L[i]
        diffs = [R[i] - L[i] for i in range(N)]
        
        # Use accumulate to find the prefix sum of the available headroom
        # pref[i] is the total headroom available from index 0 to i-1
        pref = [0] + list(accumulate(diffs))
        
        # For element i, we can add:
        # min(diffs[i], rem - (total headroom used by elements 0 to i-1))
        # However, a simpler way: 
        # The amount added to X_i is:
        # min(diffs[i], max(0, rem - pref[i]))
        # Wait, the logic is: we take as much as possible from the remainder
        # until the remainder is exhausted.
        # The amount added to X_i is:
        # min(diffs[i], rem - (pref[i+1] - diffs[i]) if rem > (pref[i+1] - diffs[i]) else 0)
        # More simply: the total added to the first i elements is min(rem, pref[i+1])
        # So X_i = L[i] + (min(rem, pref[i+1]) - min(rem, pref[i]))
        
        X = [L[i] + (min(rem, pref[i+1]) - min(rem, pref[i])) for i in range(N)]
        
        sys.stdout.write("Yes\n")
        sys.stdout.write(" ".join(map(str, X)) + "\n")
    else:
        sys.stdout.write("No\n")

if __name__ == "__main__":
    solve()