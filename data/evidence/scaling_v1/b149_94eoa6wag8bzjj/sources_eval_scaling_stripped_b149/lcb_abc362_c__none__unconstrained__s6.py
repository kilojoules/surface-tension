import sys
from itertools import accumulate

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and the pairs
    # Since we cannot use loops, we use next() to get N
    # and then group the remaining data into pairs
    data_iter = iter(input_data)
    try:
        n = next(data_iter)
    except StopIteration:
        return

    # Create lists of Ls and Rs using zip and slicing
    # We use a list comprehension to pair the remaining data
    pairs = [(L, R) for L, R in zip(data_iter, data_iter)]
    ls = [p[0] for p in pairs]
    rs = [p[1] for p in pairs]

    # Calculate the minimum and maximum possible sums
    min_sum = sum(ls)
    max_sum = sum(rs)

    # Check if 0 is within the reachable range [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We need to distribute the 'deficit' (0 - min_sum) across the intervals
    # Each X_i starts at L_i and can be increased by up to (R_i - L_i)
    deficit = -min_sum
    capacities = [r - l for l, r in zip(ls, rs)]
    
    # Use accumulate to find the prefix sums of the capacities
    # prefix_sums[i] is the total amount we could have added up to index i
    prefix_sums = list(accumulate(capacities))
    
    # For each i, the amount added to L_i is:
    # min(capacity_i, deficit - sum_of_previous_capacities)
    # This is equivalent to: min(R_i, L_i + (deficit - prefix_sum_{i-1}))
    # More simply: X_i = min(R_i, L_i + max(0, deficit - prefix_sum_{i-1}))
    # But we must ensure the total sum is exactly 0.
    
    # Let S_i be the prefix sum of capacities.
    # The amount added to X_i is min(capacity_i, max(0, deficit - S_{i-1}))
    # We can calculate this using a list comprehension.
    # We prepend 0 to prefix_sums to handle the i=0 case easily.
    s = [0] + prefix_sums
    
    # X_i = L_i + amount_added
    # amount_added = min(capacity_i, max(0, deficit - s[i]))
    # Note: s[i] is the sum of capacities BEFORE index i (because of the prepended 0)
    res = [
        ls[i] + min(capacities[i], max(0, deficit - s[i]))
        for i in range(n)
    ]

    print("Yes")
    print(*(res))

if __name__ == "__main__":
    solve()