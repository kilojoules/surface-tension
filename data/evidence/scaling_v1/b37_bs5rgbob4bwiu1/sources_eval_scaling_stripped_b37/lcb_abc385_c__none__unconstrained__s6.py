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
    # range(N) provides the indices, H[i] is the key
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
    # as the start and second element, then check how many subsequent 
    # elements exist in the set of indices for that height.
    
    # However, a more efficient way for N=3000 is to iterate over all possible
    # intervals 'd' (1 to N) and for each 'd', check the maximum length.
    # But the constraint is that they must have the SAME height.
    
    # Let's refine: For every pair of indices (i, j) with the same height,
    # they define a difference d = j - i. We check how many k exist such that
    # index i + k*d also has that same height.
    
    # To optimize, we only iterate over heights that appear multiple times.
    # For heights that appear once, the answer is 1.
    
    # We use a set for O(1) lookup of indices for each height.
    indices_sets = {h: set(idx_list) for h, idx_list in groups.items()}
    
    # Function to calculate max length for a specific height
    def get_max_for_height(h):
        idxs = sorted(groups[h])
        n_h = len(idxs)
        if n_h <= 2:
            return n_h
        
        # Try every pair of indices as the first two elements of the sequence
        # We use a set of (start_index, difference) to avoid redundant checks
        # But since we need the maximum, we can just iterate:
        res = 2
        # For every pair of indices in the sorted list
        for i in range(n_h):
            for j in range(i + 1, n_h):
                d = idxs[j] - idxs[i]
                # Check if this progression was already covered by a smaller start index
                # If (idxs[i] - d) is in the set, this is not the start of the sequence
                if (idxs[i] - d) in indices_sets[h]:
                    continue
                
                # Count how many elements are in the progression
                # We start from 2 because we already have idxs[i] and idxs[j]
                count = 2
                next_idx = idxs[j] + d
                while next_idx in indices_sets[h]:
                    count += 1
                    next_idx += d
                
                if count > res:
                    res = count
        return res

    # Map the get_max_for_height function over all unique heights
    # We use map() and max() to find the overall maximum
    ans = max(map(get_max_for_height, groups.keys()), default=0)
    
    # If N > 0 but no groups were processed, the answer should be 1 if N >= 1
    # But the constraints say N >= 1, so ans will be at least 1.
    print(ans if N > 0 else 0)

if __name__ == "__main__":
    solve()