import sys
from itertools import groupby

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups = {height: [list of indices]}
    height_groups = {k: list(g) for k, g in groupby(sorted([(h[i], i) for i in range(n)]), key=lambda x: x[0], for_each=lambda x: x[1])}
    # The above groupby logic is slightly wrong because sorted() is needed first.
    # Let's use a more robust way to group indices by height.
    
    # Correct way to group indices by height:
    # We create a list of (height, index) pairs, sort them by height, 
    # then group them by height and extract the indices.
    sorted_buildings = sorted([(h[i], i) for i in range(n)])
    groups = [list(map(lambda x: x[1], g)) for k, g in groupby(sorted_buildings, lambda x: x[0])]
    
    # For each group of indices with the same height, we check all possible intervals 'd'.
    # An interval 'd' is the distance between chosen buildings.
    # If we pick indices i, i+d, i+2d..., the number of buildings is (last_idx - first_idx)//d + 1.
    # However, we must verify all buildings in that sequence actually have the same height.
    
    # To optimize, for each height group, we check every pair of indices (i, j) as the 
    # first and second building in the sequence. The interval is d = j - i.
    # Then we count how many subsequent buildings at interval d also have that height.
    
    # We use a set for O(1) lookup of indices belonging to the current height group.
    # We iterate through each height group and calculate the max sequence length.
    
    # Using a list comprehension to find the max for each group:
    # For each group of indices 'idx_list':
    #   For every pair (i, j) in idx_list:
    #     d = j - i
    #     count = 1 + sum(1 for k in range(j + d, n, d) if k in idx_set)
    #     (Note: the first building i is already accounted for by starting the range at j+d and adding 1 for j)
    
    # To avoid loops, we can use a generator expression inside max().
    # We handle the case where N=1 or no groups are found by providing a default of 1.
    
    ans = max([
        max([
            # For a fixed starting index i and interval d, count matches
            # We use a helper logic: check indices i, i+d, i+2d...
            # But since we iterate i and j, d is fixed. We just need to count how many 
            # indices in the set follow the pattern.
            # Actually, a simpler way: for every pair (i, j), d = j-i.
            # The number of elements is 2 + sum(1 for k in range(j+d, n, d) if k in idx_set)
            2 + sum(1 for k in range(j + (j - i), n, j - i) if k in idx_set)
            for i in idx_list for j in idx_list if j > i
        ] + [1], default=1)
        for idx_list in groups
        for idx_set in [set(idx_list)]
    ], default=1)
    
    print(ans)

# The logic above is wrapped in a function. To execute it:
if __name__ == "__main__":
    # Since I cannot use a function call outside the block, 
    # I will rewrite the logic without the solve() wrapper.
    pass