import sys

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
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L)
    max_sum = sum(R)
    
    # The condition for a solution to exist is that 0 must be within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with the minimum possible values.
    # We need to increase the total sum from min_sum to 0.
    # The total amount we need to add is -min_sum.
    needed = -min_sum
    
    # For each i, the maximum we can add to L[i] is (R[i] - L[i]).
    # We use a greedy approach: fill each X_i to its maximum until 'needed' is exhausted.
    
    # Calculate the headroom for each element
    headroom = [r - l for l, r in zip(L, R)]
    
    # We need to distribute 'needed' across the headrooms.
    # The amount added to X_i is min(headroom[i], remaining_needed).
    # Since we cannot use loops, we can use a trick with itertools.accumulate 
    # to find how much of the 'needed' sum is consumed by the time we reach index i.
    
    import itertools
    
    # cumulative_headroom[i] is the sum of headroom[0...i-1]
    cum_headroom = list(itertools.accumulate(headroom))
    
    # The amount added to X_i is:
    # max(0, min(headroom[i], needed - (sum of headrooms before i)))
    # Let S_{i-1} be the sum of headrooms before index i.
    # Added amount = max(0, min(headroom[i], needed - S_{i-1}))
    
    # To implement this without a loop, we calculate S_{i-1} using the accumulated list.
    # S_{i-1} = cum_headroom[i-1] if i > 0 else 0.
    
    # We can use a list comprehension to calculate the final X values.
    # Note: cum_headroom[i-1] is used to determine if the 'needed' quota is already filled.
    
    # To handle the index 0 case for cum_headroom, we can prepend 0.
    s_prev = [0] + cum_headroom[:-1]
    
    # X_i = L_i + amount_added
    # amount_added = max(0, min(headroom[i], needed - s_prev[i]))
    # However, if needed - s_prev[i] is negative, it means the quota was filled earlier.
    
    X = [l + max(0, min(h, needed - s)) for l, h, s in zip(L, headroom, s_prev)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()