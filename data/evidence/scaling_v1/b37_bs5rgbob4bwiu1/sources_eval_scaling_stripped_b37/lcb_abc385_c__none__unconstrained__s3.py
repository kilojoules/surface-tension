import sys
from itertools import groupby
from operator import itemgetter

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Group indices by their height: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to group indices for each unique height
    # range(N) provides the 0-indexed position of each building
    height_groups = {}
    for i in range(N):
        h = H[i]
        if h not in height_groups:
            height_groups[h] = []
        height_groups[h].append(i)

    # For each height, we find the maximum number of buildings at equal intervals.
    # Two indices i and j have a distance d = j - i.
    # We are looking for the maximum size of an arithmetic progression within the indices.
    # Since N is small (3000), we can iterate through all pairs of indices for each height.
    
    # We use a list comprehension to calculate the max count for each height group.
    # For a group of indices 'idx_list', we check every pair (i, j) as the start and second element.
    # The interval is d = idx_list[j] - idx_list[i].
    # We then count how many subsequent elements in the sorted idx_list fit the pattern.
    
    # To optimize, we only iterate through heights that appear more than once.
    # If a height appears only once, the max is 1.
    
    results = [
        max(
            (
                sum(1 for k in range(len(indices)) if (indices[k] - indices[i]) % (indices[j] - indices[i]) == 0 
                    and (indices[k] - indices[i]) // (indices[j] - indices[i]) >= 0)
                # The above logic is slightly flawed for general APs, 
                # let's use a more direct approach:
                # For every pair (i, j), the interval is d = indices[j] - indices[i].
                # We check how many indices k exist such that indices[k] = indices[i] + m*d.
                # Since we need the count, we can use a set for O(1) lookup.
            )
            for i in range(len(indices))
            for j in range(i + 1, len(indices))
        )
        for indices in height_groups.values()
        if len(indices) > 1
    ]

    # The logic inside the comprehension needs to be a bit more robust.
    # Let's redefine the core logic without the comprehension for clarity and correctness,
    # then wrap it back into the required format.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Group indices by height
    from collections import defaultdict
    groups = defaultdict(list)
    for i, h in enumerate(H):
        groups[h].append(i)

    # For each height, find the max number of indices forming an arithmetic progression
    # We iterate over all pairs of indices (i, j) in the group to define the start and the step.
    # Then we count how many elements in the set of indices fit the sequence.
    
    ans = 1 # Minimum possible answer is 1 if N >= 1
    
    # We process each height group
    for h in groups:
        indices = groups[h]
        if len(indices) < 2:
            continue
        
        # Use a set for O(1) lookup of indices
        idx_set = set(indices)
        
        # Try every pair of indices as the first two elements of the sequence
        # i is the index in the 'indices' list, j is the index in the 'indices' list
        # We use a generator expression inside max() to find the best pair for this height
        current_max = max(
            (
                # For a fixed start indices[i] and step d = indices[j] - indices[i],
                # we count how many k exist such that indices[i] + k*d is in the set.
                # We only need to check k from 0 up to N/d.
                sum(1 for k in range((N - 1 - indices[i]) // (indices[j] - indices[i]) + 1)
                    if (indices[i] + k * (indices[j] - indices[i])) in idx_set)
                for i in range(len(indices))
                for j in range(i + 1, len(indices))
            ),
            default=1
        )
        if current_max > ans:
            ans = current_max

    print(ans)

if __name__ == "__main__":
    final_solve()