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

    # Group indices by height: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to group indices for each unique height
    # range(N) provides the indices, H[i] provides the height
    groups = {}
    for i in range(N):
        h = H[i]
        if h not in groups:
            groups[h] = []
        groups[h].append(i)

    # For each height group, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices
    # within each height group to define a starting point and a common difference.
    
    # However, a more efficient way to check "equal intervals" for a fixed height:
    # For every pair of indices (i, j) in the group, the difference d = j - i.
    # We check how many elements in the group fit the pattern i, i+d, i+2d...
    
    # To optimize, we process each height group separately.
    # For a group of indices 'indices', we try all pairs (indices[i], indices[j])
    # as the first two elements of the sequence.
    
    # Using a set for O(1) lookup of indices within a height group.
    results = [
        max(
            (
                # For every pair of indices in the group, calculate the difference d.
                # Then calculate how many terms of the sequence exist in the set.
                # The number of terms is (max_index - start_index) // d + 1,
                # provided the last term actually exists in the set.
                # Actually, the simplest way is to count how many k satisfy (start + k*d) in index_set.
                # But we can just iterate k until the index exceeds N.
                len([k for k in range(0, N, d) if (start + k) in index_set])
                for start, indices_list in [ (idx, indices) for idx in indices ]
                for d in [indices_j - start for indices_j in indices if indices_j > start]
            )
            if indices else [1]
        )
        for height, indices in groups.items()
        for index_set in [set(indices)]
        for indices in [indices]
    ]

    # The above comprehension is a bit complex, let's refine it into a 
    # more readable logic while keeping it inside the solve function.
    
    # We wrap the logic in a helper-like comprehension:
    # For each height, we find the max length of an arithmetic progression.
    # If a height appears only once, the length is 1.
    # If it appears multiple times, we check all pairs.
    
    ans = max([
        max([
            # For each pair of indices (i, j) in the sorted list of indices for a height:
            # i is the start, j-i is the difference d.
            # We count how many elements in the set follow the pattern.
            sum(1 for k in range(i, N, j - i) if k in index_set)
            for idx_i, i in enumerate(indices)
            for j in indices[idx_i + 1:]
        ] + [1])
        for height, indices in groups.items()
        for index_set in [set(indices)]
    ])

    print(ans)

if __name__ == "__main__":
    solve()