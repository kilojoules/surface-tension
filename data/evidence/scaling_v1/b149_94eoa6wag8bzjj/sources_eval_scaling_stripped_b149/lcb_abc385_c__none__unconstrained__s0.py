import sys
from collections import defaultdict

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_groups = {height: [index1, index2, ...]}
    height_groups = defaultdict(list)
    for i in range(n):
        height_groups[h[i]].append(i)
    
    # For each height, we want to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since we need to check all possible intervals 'd', 
    # we can iterate through all pairs of indices (i, j) within the same height group.
    
    # We use a list comprehension to evaluate the maximum for each height group.
    # For a fixed height, we try every pair of indices (i, j) as the first two elements.
    # The interval is d = j - i. We then count how many subsequent elements k = j + d, j + 2d...
    # also have that same height.
    
    # To avoid loops, we use a generator expression inside max().
    # We iterate over each height's list of indices.
    
    results = [
        max(
            (
                # For every pair of indices (idx1, idx2) in the group
                # count how many elements in the sequence idx1, idx1+d, ... 
                # are present in the set of indices for that height.
                # d = idx2 - idx1
                # The number of elements is the count of k such that 
                # (idx1 + k*d) < n and h[idx1 + k*d] == height
                # However, a simpler way is to just check the sequence 
                # and stop when the height doesn't match.
                # Since we can't use while, we can use a range and a trick 
                # with itertools.takewhile or just a list comprehension 
                # checking the condition.
                
                # Let's use a more direct approach: 
                # For a fixed start i and interval d, the count is:
                # sum(1 for k in range(0, n, d) if i + k < n and h[i + k] == height)
                # But we must iterate over all possible d.
                0 # Placeholder for the logic inside the comprehension
            )
            for height, indices in height_groups.items()
        )
    ]
    
    # Correcting the logic to fit inside a single expression:
    # We iterate over all unique heights.
    # For each height, we iterate over all possible starting positions 'i' 
    # and all possible intervals 'd' (from 1 to N).
    
    ans = max(
        (
            max(
                (
                    sum(1 for k in range(i, n, d) if h[k] == height)
                    for i in range(n) if h[i] == height
                    for d in range(1, n)
                ) + 1 if not height_groups else 1
            )
            for height in height_groups.keys()
        ),
        default=0
    )
    
    # The above logic is slightly flawed because it checks all d for every i.
    # Let's refine: for every height, and every pair of indices (i, j) in that height's group,
    # we treat i as the start and d = j - i as the interval.
    
    final_ans = max(
        (
            max(
                (
                    sum(1 for k in range(i, n, j - i) if h[k] == height)
                    for idx_i, i in enumerate(indices)
                    for j in indices[idx_i + 1:]
                ),
                default=1
            )
            for height, indices in height_groups.items()
        ),
        default=0
    )
    
    print(final_ans)

# Since the prompt requires a complete program and the logic above is 
# wrapped in a function, I will call it. 
# However, the prompt asks for the code block. 
# I will rewrite the logic into a single executable script.

if __name__ == "__main__":
    # Re-implementing the logic cleanly without the function wrapper for the final output
    import sys
    from collections import defaultdict

    input_data = sys.stdin.read().split()
    if input_data:
        n = int(input_data[0])
        h = list(map(int, input_data[1:]))
        height_groups = defaultdict(list)
        for i in range(n):
            height_groups[h[i]].append(i)

        print(max(
            (
                max(
                    (
                        sum(1 for k in range(i, n, j - i) if h[k] == height)
                        for idx_i, i in enumerate(indices)
                        for j in indices[idx_i + 1:]
                    ),
                    default=1
                )
                for height, indices in height_groups.items()
            ),
            default=0
        ))