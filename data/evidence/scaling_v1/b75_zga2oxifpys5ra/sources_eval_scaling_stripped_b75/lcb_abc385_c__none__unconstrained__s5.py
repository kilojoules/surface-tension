import sys
from itertools import groupby

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # Group indices by their building height
    # height_groups will be a dictionary where keys are heights 
    # and values are lists of indices where that height occurs.
    # Example: {5: [0, 2, 5], 7: [1, 3, 4, 6, 7]}
    from collections import defaultdict
    height_groups = defaultdict(list)
    for idx, height in enumerate(h):
        height_groups[height].append(idx)

    # For a fixed height, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices
    # (i, j) for each height to define a starting point and a common difference.
    
    # We use a comprehension to find the max for each height group.
    # For each pair of indices (start, next_idx) in a group:
    # diff = next_idx - start
    # We count how many elements in that group fit the pattern: start + k*diff
    # However, a simpler approach for N=3000 is to iterate through all possible
    # differences 'd' from 1 to N, and all starting positions 's' from 0 to N-1.
    
    # Let's use a more optimized approach:
    # For each height, and for each pair of indices in that height's group,
    # check the length of the sequence.
    
    # To avoid O(N^3), we can't just loop. But we can iterate through 
    # all possible differences d in [1, N].
    # For a fixed d, we can check all starting positions s in [0, N-1].
    # But that's still slow. 
    
    # Actually, the most efficient way given the constraints is:
    # For each height, get the list of indices.
    # For every pair of indices (i, j) in that list, they define a difference d = j - i.
    # We can then check how many subsequent indices (j + d, j + 2d...) also have that height.
    
    # Since we need the maximum, we can use a generator expression inside max().
    # We handle the case where only 1 building is chosen by initializing max with 1.
    
    # Optimization: Only iterate through pairs in the same height group.
    # For a fixed height group 'indices', and a pair indices[i], indices[j]:
    # the difference is d = indices[j] - indices[i].
    # We count k such that indices[i] + k*d is in the set of indices.
    
    # Using a set for O(1) lookup.
    res = max([
        max([
            sum(1 for k in range((len(indices) - 1) // 1 + 1) 
                if (indices[i] + k * (indices[j] - indices[i])) in indices_set))
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]) if len(indices) > 1 else 1
        for indices in height_groups.values()
        for indices_set in [set(indices)]
    ], default=1)

    # The above logic is slightly flawed in the range of k. 
    # Let's refine: for a fixed start s and difference d, 
    # the number of elements is (last_index - s)//d + 1, 
    # provided all intermediate steps are present.
    
    # Corrected logic using a helper to count:
    def count_sequence(s, d, s_set):
        # We don't know the end, so we increment until the element is not in the set
        # But we can't use loops, so we use a trick with range and next()
        # We find the first k where (s + k*d) is not in s_set
        # The number of elements is that k.
        return next((k for k in range(1, n + 1) if (s + k*d) not in s_set), n)

    # Re-calculating the result without loops:
    final_res = max([
        max([
            count_sequence(indices[i], indices[j] - indices[i], indices_set)
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        ]) if len(indices) > 1 else 1
        for indices in height_groups.values()
        for indices_set in [set(indices)]
    ], default=1)

    print(final_res)

if __name__ == "__main__":
    solve()