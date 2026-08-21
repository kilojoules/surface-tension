import sys
from itertools import groupby
from operator import itemgetter

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups will be a dictionary where keys are heights 
    # and values are sorted lists of indices where that height occurs.
    # Example: {5: [0, 2, 5], 7: [1, 3, 4, 6, 7]}
    indexed_heights = sorted([(H[i], i) for i in range(N)], key=itemgetter(0))
    height_groups = [list(map(itemgetter(1), g)) 
                     for k, g in groupby(indexed_heights, key=itemgetter(0))]
    
    # For each height group, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need equal intervals, we are looking for indices i, i+d, i+2d...
    # For a fixed height, we can iterate through all pairs of indices (i, j)
    # as the start and second element, calculating the common difference d = j - i.
    # Then we count how many subsequent elements (j+d, j+2d...) also have that height.
    
    # To optimize, we use a set for O(1) lookup of indices for each height.
    results = [
        max(
            (len([i for i in range(start, N, diff) if i in index_set])
             for diff in range(1, N) 
             for start in sorted_indices)
            if len(sorted_indices) > 1 else [1],
            default=1
        )
        for sorted_indices in height_groups
        for index_set in [set(sorted_indices)]
    ]
    
    # The above comprehension is a bit dense, let's refine the logic into a 
    # more readable structure while keeping it within the constraints.
    # We use a helper function inside a list comprehension to process each height group.
    
    def get_max_for_group(indices):
        if len(indices) <= 2:
            return len(indices)
        
        s = set(indices)
        # Try every pair of indices as the first two elements of the sequence
        # diff = indices[j] - indices[i]
        # We only need to check pairs (i, j) where i < j
        # For each pair, we count how many terms exist in the set.
        return max(
            (sum(1 for k in range(indices[i], N, indices[j] - indices[i]) if k in s)
             for i in range(len(indices))
             for j in range(i + 1, len(indices))),
            default=1
        )

    # Map the helper over all height groups and find the global maximum
    final_ans = max(get_max_for_group(group) for group in height_groups)
    print(final_ans)

if __name__ == "__main__":
    solve()