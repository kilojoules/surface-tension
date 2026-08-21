import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs using slicing and map
    L = list(map(int, input_data[1::2]))
    R = list(map(int, input_data[2::2]))
    
    # Calculate the minimum possible sum and the maximum possible sum
    sum_L = sum(L)
    sum_R = sum(R)
    
    # The condition for a solution to exist is that 0 must fall within [sum_L, sum_R]
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We need to add 'diff' to the sum of Ls to reach 0
    diff = -sum_L
    
    # Headroom for each X_i is R_i - L_i
    headroom = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find the prefix sum of headrooms
    # This allows us to determine how much of the 'diff' is absorbed by each element
    prefix_headroom = list(accumulate(headroom))
    
    # For each i, the amount added to L_i is:
    # min(headroom[i], diff - (prefix_headroom[i-1] if i > 0 else 0))
    # However, a simpler way is:
    # The total added to X_i is the portion of 'diff' that falls within its headroom range.
    # Amount added to X_i = max(0, min(headroom[i], diff - (prefix_headroom[i-1] if i > 0 else 0)))
    
    # To avoid if/else in comprehension, we can prepend a 0 to prefix_headroom
    ph = [0] + prefix_headroom
    
    # X_i = L_i + amount_contributed
    # The amount contributed by index i is the intersection of [ph[i], ph[i+1]] and [0, diff]
    # Which is max(0, min(ph[i+1], diff) - ph[i])
    X = [L[i] + max(0, min(ph[i+1], diff) - ph[i]) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()