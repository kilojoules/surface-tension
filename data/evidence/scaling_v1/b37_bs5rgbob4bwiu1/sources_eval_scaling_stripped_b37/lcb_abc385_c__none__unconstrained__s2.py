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
    # range(N) provides the indices, H[i] provides the height
    groups = {}
    for i in range(N):
        h = H[i]
        if h not in groups:
            groups[h] = []
        groups[h].append(i)

    # For each height group, we want to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need equal intervals, we are looking for indices i, i+d, i+2d...
    # For a fixed height, we can iterate through all pairs of indices (i, j)
    # as the starting point and the second point, then check how many 
    # subsequent points (j + (j-i), ...) also have that same height.
    
    # However, a more efficient way is to iterate over all possible intervals 'd'
    # and all possible starting positions 's'.
    # But since N is small (3000), we can iterate over each unique height,
    # and for that height, iterate over all pairs of indices to define the interval.
    
    # To optimize: for a specific height, if there are 'k' buildings,
    # we check all pairs. If k is small, this is fast. If k is large, 
    # we still only check pairs.
    
    # Let's use a helper function to calculate the max for a single height group
    def max_for_height(indices):
        n_idx = len(indices)
        if n_idx <= 2:
            return n_idx
        
        # We use a set for O(1) lookup of indices
        idx_set = set(indices)
        
        # For every pair of indices (i, j), they define a difference d = j - i.
        # We count how many elements exist in the sequence i, i+d, i+2d...
        # To avoid redundant checks, we only consider i < j.
        # We use a comprehension to calculate the length for every pair and find the max.
        # We use a generator expression inside max() to keep memory low.
        
        # Optimization: if we find a sequence of length L, and the remaining 
        # indices in the group are fewer than L, we can't do better. 
        # But the comprehension approach is cleaner for "pure" Python.
        
        return max(
            (
                sum(1 for k in range(i + 1, N, j - i) if k in idx_set) + 1
                for i in range(n_idx)
                for j in range(i + 1, n_idx)
            ),
            default=1
        )

    # We apply the logic to every group of indices and take the overall maximum.
    # We use a generator to process each height group.
    ans = max(
        (max_for_height(indices) for indices in groups.values()),
        default=0
    )
    
    # The logic above for max_for_height handles the "range" logic.
    # Wait, the range logic `range(i + 1, N, j - i)` uses the value of the index,
    # not the position in the `indices` list. Let's refine that.
    
    # Corrected logic inside a function:
    def solve_refined():
        # Using the logic: for every pair of indices (idx1, idx2) in the sorted 
        # list of indices for a height, the difference is d = idx2 - idx1.
        # We count how many idx1 + m*d are also in the set.
        
        # We can use a set comprehension to find the max for each height group.
        # Since we cannot use loops, we use a nested generator.
        
        # For each height group:
        # 1. Get sorted indices.
        # 2. For every pair (indices[i], indices[j]), calculate length.
        # 3. The length is (number of k such that indices[i] + k*(indices[j]-indices[i]) is in set)
        
        # To avoid the range() issue with indices, we use:
        # (indices[j] - indices[i]) as the step.
        # We check how many steps we can take before exceeding N.
        
        # Since we need to return a single block, I will integrate this into the main flow.
        pass

# Redefining the logic to fit the "no for-loop" constraint strictly using comprehensions
def final_solve():
    input_data = sys(stdin := sys.stdin).read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Group indices by height
    # {height: {index1, index2, ...}}
    groups = {h: {i for i, x in enumerate(H) if x == h} for h in set(H)}
    
    # For each height, for every pair of indices (i, j) in that height's set:
    # Calculate the length of the arithmetic progression.
    # We use a generator to iterate over heights, then pairs, then counts.
    
    # To avoid nested loops, we use nested generator expressions.
    # For a fixed height group 'S', and two indices i, j in S (i < j):
    # The difference is d = j - i.
    # The number of elements is the count of k >= 0 such that i + k*d is in S.
    # Since i + k*d < N, k < (N - i) / d.
    
    res = max(
        (
            max(
                (
                    sum(1 for k in range(i, N, j - i) if k in S)
                    for i in S
                    for j in S if j > i
                ),
                default=1
            )
            for S in groups.values()
        ),
        default=0
    )
    print(res)

if __name__ == "__main__":
    final_solve()