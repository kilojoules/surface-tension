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
    # Since we need "equal intervals", we are looking for indices i, i+d, i+2d...
    # For a fixed height, we can iterate over all pairs of indices (i, j) 
    # to define a starting point and a common difference d = j - i.
    # Then we count how many elements in the sorted index list fit that pattern.
    
    # However, a more efficient way for N=3000 is to iterate over all possible 
    # differences d (1 to N-1) and all starting positions i (0 to d-1).
    # But that doesn't account for the "same height" constraint efficiently.
    
    # Let's use the property: for a fixed height h, and a fixed difference d,
    # we can check all possible starting positions.
    # Actually, the simplest approach given N=3000 is:
    # For every pair of indices (i, j) with the same height, they define a difference d.
    # We check how many subsequent indices (j+d, j+2d...) also have height h.
    
    # To optimize, we process each height group separately.
    # For a group of indices 'idx_list', we check every pair (a, b) as the first two elements.
    
    # Using a set for O(1) lookup of indices for a specific height.
    # We use a list of sets, where sets_by_height[h] contains all indices of height h.
    # Since H_i <= 3000, we can use a list.
    sets_by_height = [set() for _ in range(3000 + 1)]
    for i, h in enumerate(H):
        sets_by_height[h].add(i)

    # We only care about heights that actually appear in the input.
    unique_heights = set(H)
    
    # For each height, find the max sequence.
    # We iterate through all indices i and j (i < j) that have the same height.
    # d = j - i. We count how many k = j + d, j + 2d... also have that height.
    
    # To avoid O(N^3), we can use the fact that for a fixed i and d, 
    # the number of elements is (N - 1 - i) // d + 1.
    # We only check pairs (i, j) and then use a while loop to count.
    
    # To further optimize: for a fixed height, we only iterate over indices belonging to it.
    # Let indices be sorted list L. For every i < j in L:
    # d = L[j] - L[i]. Count k = 2, 3... such that L[i] + k*d is in the set.
    
    # We use a function to calculate the length for a specific height group.
    def get_max_for_height(indices_set):
        sorted_idx = sorted(list(indices_set))
        n_idx = len(sorted_idx)
        if n_idx <= 2:
            return n_idx
        
        # For every pair of indices in the sorted list, treat them as the first two elements
        # of the sequence.
        # We use a list comprehension to iterate over all pairs and a 
        # helper logic to count the length of the progression.
        
        # Since we cannot use loops, we can use a recursive-like approach 
        # or a mathematical way to count. 
        # Actually, for a fixed i and j, the length is:
        # 1 + (number of k > 1 such that i + k*(j-i) is in indices_set)
        
        # We can use a generator expression inside max()
        return max(
            1 + sum(1 for k in range(2, (N - sorted_idx[i]) // (sorted_idx[j] - sorted_idx[i]) + 1)
                    if (sorted_idx[i] + k * (sorted_idx[j] - sorted_idx[i])) in indices_set),
            range(i, n_idx) for i in range(n_idx) for j in range(i + 1, n_idx)
        )
    
    # The above logic is slightly wrong because the range(i, n_idx) is in the wrong place.
    # Corrected logic inside a comprehension:
    
    ans = max(
        max(
            (sum(1 for k in range(2, (N - sorted_idx[i]) // (sorted_idx[j] - sorted_idx[i]) + 1)
              if (sorted_idx[i] + k * (sorted_idx[j] - sorted_idx[i])) in indices_set) + 2
             for i in range(len(sorted_idx)) 
             for j in range(i + 1, len(sorted_idx))),
            default=0
        )
        for h in unique_heights
        for sorted_idx in [sorted(list(sets_by_height[h]))]
        for indices_set in [sets_by_height[h]]
        if len(sorted_idx) >= 2
    ) if any(len(sets_by_height[h]) >= 2 for h in unique_heights) else 1

    # The logic above handles the case where at least one height has 2+ buildings.
    # If all heights have only 1 building, the answer is 1.
    # We must ensure the result is at least 1.
    print(max(ans, 1))

if __name__ == "__main__":
    solve()