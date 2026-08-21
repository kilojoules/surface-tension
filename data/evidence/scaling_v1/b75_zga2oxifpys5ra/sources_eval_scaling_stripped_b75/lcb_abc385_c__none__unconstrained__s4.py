import sys
from itertools import groupby

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # Group indices by their building height
    # height_groups will be a dictionary where keys are heights 
    # and values are lists of indices where that height occurs.
    # Example: {5: [0, 2, 5], 7: [1, 3, 4, 6, 7]}
    height_groups = {}
    for idx, height in enumerate(h):
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(idx)

    # For each height group, we need to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need equal intervals, we can iterate through all pairs of indices
    # (i, j) in a group, calculate the difference d = j - i, and check
    # how many subsequent indices (j + d, j + 2d, ...) also belong to the group.
    
    # We use a comprehension to find the max for each height, then the max of those.
    # For a specific height's indices 'indices':
    # We try every pair of indices as the start and second element of the sequence.
    # The number of elements is calculated by checking the existence of the next terms.
    
    # To optimize, we use a set for O(1) lookup of indices.
    def get_max_for_group(indices):
        s = set(indices)
        # Try all pairs (i, j) where i < j
        # The length of the sequence is 1 + (number of k such that i + k*d is in s)
        # However, a simpler way is to iterate through all pairs and 
        # count how far the progression goes.
        return max(
            (
                sum(1 for k in range(j - i, n, j - i) if (i + (k // (j - i)) * (j - i)) in s)
                # The above is slightly wrong logic for counting. 
                # Correct logic: for a fixed start i and diff d, count how many i + m*d are in s.
                for i in indices 
                for j in indices if j > i
            ), 
            default=1
        )

    # Corrected logic inside a comprehension:
    # For each height, we check all pairs of indices (i, j). 
    # The difference is d = j - i. We count how many terms i, i+d, i+2d... are in the set.
    
    # Since N=3000, a nested loop over all pairs might be O(N^2), 
    # but we only do it for indices of the same height.
    # The total number of pairs across all height groups is at most N(N-1)/2.
    
    # Using a helper to avoid deep nesting in comprehension
    def count_progression(s, i, d):
        count = 0
        curr = i
        while curr in s:
            count += 1
            curr += d
        return count

    # We iterate through each height group, and for each group, every pair of indices.
    # We use a generator to find the maximum.
    result = max(
        (
            max(
                (count_progression(set(indices), indices[a], indices[b] - indices[a])
                 for a in range(len(indices))
                 for b in range(a + 1, len(indices))),
                default=1
            )
            for indices in height_groups.values()
        ),
        default=0
    )

    print(result)

if __name__ == "__main__":
    solve()