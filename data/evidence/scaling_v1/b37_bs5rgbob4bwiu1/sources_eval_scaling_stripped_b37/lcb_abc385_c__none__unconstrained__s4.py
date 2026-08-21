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
    # range(N) provides the index, H[i] provides the height
    groups = {}
    for i in range(N):
        h = H[i]
        if h not in groups:
            groups[h] = []
        groups[h].append(i)

    # For each height group, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need "equal intervals", we are looking for indices i, i+d, i+2d...
    # For a fixed height, we check every pair of indices (i, j) as the start and second element.
    # The common difference is d = j - i. 
    # The number of elements isK = (last_index - first_index) // d + 1
    # provided that all elements in the sequence exist in the set of indices for that height.
    
    # To optimize, we iterate through each height group and check all possible differences.
    # For a sorted list of indices 'idx_list', and a difference 'd', 
    # we can use a set for O(1) lookup.
    
    results = [
        max(
            (
                len([i for i in range(start, N, d) if i in idx_set])
                for start in idx_list
                for d in range(1, N // 1 + 1) # d can range from 1 to N
                if (start + d) in idx_set
            ) or [1], 
            default=[1]
        )
        for idx_list in groups.values()
        for idx_set in [set(idx_list)]
    ]

    # The above comprehension is a bit complex, let's refine it into a helper function
    # and use a map/list comprehension to find the maximum across all height groups.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Group indices by height
    # height_map = {h: {indices}}
    height_map = {}
    for i, h in enumerate(H):
        if h not in height_map:
            height_map[h] = set()
        height_map[h].add(i)

    # For each height, find the max sequence length
    # We iterate over all pairs of indices (i, j) in the set to define the interval d = j - i
    # Then we count how many k exist such that i + k*d is also in the set.
    
    def get_max_for_height(indices_set):
        idx_list = sorted(list(indices_set))
        n_idx = len(idx_list)
        if n_idx <= 2:
            return n_idx
        
        # Try every pair of indices as the first two elements of the sequence
        # i is the start index, j is the second index
        # Difference d = j - i
        # We use a generator expression to find the length for every pair
        return max(
            (
                # Count how many elements of the progression exist in the set
                # We use a while loop logic inside a helper or a comprehension
                # Since we cannot use while loops in comprehensions, 
                # we can use range(start, N, d) and check membership
                sum(1 for k in range(idx_list[i], N, idx_list[j] - idx_list[i]) 
                    if k in indices_set)
                for i in range(n_idx)
                for j in range(i + 1, n_idx)
            ),
            default=1
        )

    # Apply the helper to all height groups and find the global maximum
    ans = max(get_max_for_height(s) for s in height_map.values())
    print(ans)

if __name__ == "__main__":
    final_solve()