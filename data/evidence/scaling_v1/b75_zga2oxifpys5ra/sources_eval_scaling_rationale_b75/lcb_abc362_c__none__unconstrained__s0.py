import sys
from itertools import accumulate

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    it = iter(input_data)
    
    # N is the first element
    N = next(it)
    
    # Group the remaining elements into pairs of (L, R)
    # Using a list comprehension to create the pairs
    pairs = [(L := next(it), R := next(it)) for _ in range(N)]
    
    # Calculate the minimum and maximum possible sums
    # Using generator expressions inside sum()
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # The condition for a solution to exist is that 0 must fall within [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start by setting every X_i to its minimum value L_i
    # We need to add a total of 'target_diff' to the elements to reach a sum of 0
    target_diff = -min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i)
    diffs = [p[1] - p[0] for p in pairs]
    
    # We use accumulate to find how much of the target_diff is consumed as we go
    # current_total_diffs[i] is the sum of (R_j - L_j) for j from 0 to i
    current_total_diffs = list(accumulate(diffs))
    
    # For each i, the amount added to L_i is:
    # min(diffs[i], target_diff - sum_of_previous_diffs)
    # This can be expressed as: min(diffs[i], max(0, target_diff - current_total_diffs[i-1]))
    # However, a simpler way to think about it is:
    # The amount added to X_i is the portion of target_diff that falls within the range
    # [current_total_diffs[i-1], current_total_diffs[i]]
    
    # To avoid indices and loops, we can zip the accumulated sums
    # prev_totals starts at 0
    prev_totals = [0] + current_total_diffs[:-1]
    
    # The amount to add to L_i is:
    # clamp(target_diff, prev_total, current_total) - prev_total
    # Which simplifies to: min(R_i - L_i, max(0, target_diff - prev_total))
    
    X = [
        p[0] + min(d, max(0, target_diff - pt))
        for p, d, pt in zip(pairs, diffs, prev_totals)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()