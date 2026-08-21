import sys
from itertools import groupby

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Group indices by the height of the building
    # height_map: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to build the map
    # To avoid loops, we can use a technique with groupby or a custom reduce
    # But since we can't use loops, we'll use a dictionary and list comprehensions
    
    # Get all unique heights present in the input
    unique_heights = sorted(set(H))
    
    # For each unique height, find all indices where that height occurs
    # Then, for every pair of indices (i, j), calculate the interval d = j - i
    # The number of buildings in that sequence is (j - i) // d + 1 if it's a valid sequence
    # However, the condition is "equal intervals", meaning we check if 
    # indices i, i+d, i+2d... all have the same height.
    
    # Let's redefine: for every pair of indices (i, j) with the same height,
    # they could be the first and last elements of a sequence with interval d = (j-i)//(k-1)
    # But it's simpler to:
    # 1. Pick a height 'h'
    # 2. Pick two indices i and j (i < j) where H[i] == H[j] == h
    # 3. Assume they are the 1st and 2nd elements. The interval is d = j - i.
    # 4. Count how many subsequent elements H[j+d], H[j+2d]... also equal 'h'.
    
    # To avoid loops and recursion, we can use a mathematical approach:
    # For a fixed height 'h' and a fixed interval 'd', 
    # we can check all possible starting positions 's' (0 <= s < d).
    # The number of buildings is the count of indices (s + k*d) that have height 'h'.
    # This is still a search. 
    
    # Let's use the property that N is small (3000).
    # We can iterate through all possible intervals d (1 to N-1).
    # For a fixed d, we can group indices by (index % d).
    # Within each group, we look for the longest contiguous block of the same height.
    
    # Since we cannot use loops, we use map/filter/comprehensions.
    # We evaluate all d in range(1, N)
    # For each d, we evaluate all s in range(d)
    # For each (d, s), we extract the sequence H[s], H[s+d], H[s+2d]...
    # We use groupby to find the length of the longest block of identical heights.
    
    # The result is the maximum of all these lengths.
    
    ans = max(
        # For every possible interval d
        # For every possible start s
        # Group the sequence by height and find the length of the groups
        # Use a list comprehension to flatten the results
        [
            len(list(group))
            for d in range(1, N)
            for s in range(d)
            for key, group in groupby(H[s::d])
        ],
        # Default to 1 if the list is empty
        default=1
    )
    
    print(ans)

if __name__ == "__main__":
    solve()