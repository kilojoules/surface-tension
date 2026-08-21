import sys
from collections import defaultdict

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_map = {height: [index1, index2, ...]}
    height_map = defaultdict(list)
    for i in range(n):
        height_map[h[i]].append(i)
    
    # For each height, we want to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since N is up to 3000, we can iterate through all pairs of indices 
    # for each height to define a potential interval.
    
    # We use a list comprehension to iterate over all heights and their indices.
    # For each height, we check all pairs (i, j) as the first two elements 
    # of the sequence and count how many subsequent elements fit the pattern.
    
    # However, a more efficient way to check intervals is to iterate 
    # through all possible intervals 'd' for each height.
    
    # Let's refine: for each height, we have a sorted list of indices.
    # We can check every pair of indices (idx1, idx2) as the start and second element.
    # The common difference is d = idx2 - idx1.
    # We then check if idx2 + d, idx2 + 2d... are also in the set of indices.
    
    # To avoid loops, we can use a generator expression.
    # We pre-calculate the set of indices for each height for O(1) lookup.
    
    indices_sets = {height: set(idxs) for height, idxs in height_map.items()}
    
    # For each height, and for each pair of indices in that height's list:
    # We calculate the length of the progression.
    # We use a helper logic inside a comprehension.
    
    # Since we cannot use 'while' or 'for' loops, we can use a recursive-like 
    # structure via a list comprehension or use the fact that the maximum 
    # length is N. We can check if (start + k*d) exists in the set for k=0...N.
    
    # The logic: for a fixed height, start index 's', and difference 'd':
    # The number of elements is the count of k >= 0 such that (s + k*d) is in the set.
    
    # We iterate over all heights, all possible start indices 's' from the list,
    # and all possible differences 'd' (where s+d is also in the list).
    
    ans = max(
        (
            sum(1 for k in range(n) if (s + k * d) in indices_sets[height])
            for height, idxs in height_map.items()
            for i in range(len(idxs))
            for j in range(i + 1, len(idxs))
            for s, d in [(idxs[i], idxs[j] - idxs[i])]
        ),
        default=1
    )
    
    print(ans)

if __name__ == "__main__":
    solve()