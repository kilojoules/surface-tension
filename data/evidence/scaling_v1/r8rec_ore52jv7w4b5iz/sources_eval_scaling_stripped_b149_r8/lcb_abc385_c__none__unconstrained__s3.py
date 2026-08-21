import sys
from itertools import groupby

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups = {height: [list of indices]}
    # We use a dictionary comprehension to organize indices for each unique height
    unique_heights = set(h)
    height_groups = {height: [i for i, val in enumerate(h) if val == height] 
                     for height in unique_heights}
    
    # For each height group, we need to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices (i, j)
    # in a group to define a starting point and a common difference 'd'.
    
    # We use a nested comprehension to evaluate all possible intervals for each height.
    # For a fixed height and a fixed interval 'd', we count how many buildings 
    # of that height exist at that interval.
    
    # However, a simpler approach: 
    # For every pair of indices (i, j) in a height group, they define an interval d = j - i.
    # We can then check how many elements in that group fit the pattern i + k*d.
    # But that's O(N^3). With N=3000, we need something closer to O(N^2).
    
    # Optimized approach:
    # For each height, and for every pair of indices (i, j) where i < j,
    # the number of elements is (count of k such that h[i + k*(j-i)] == height).
    # Actually, we can just iterate through all possible intervals d from 1 to N.
    # For a fixed d and height H, we check all possible offsets r from 0 to d-1.
    
    # Let's use a more direct approach:
    # For each height, we find all indices. Then for every pair of indices, 
    # we calculate the potential length.
    # To avoid O(N^3), we can iterate through all possible intervals d (1 to N)
    # and all possible starting positions i (0 to N-1).
    
    # The condition is: H[i] == H[i+d] == H[i+2d] ...
    # We can use a generator expression to find the length of the sequence 
    # starting at i with interval d.
    
    # We use a helper function to count the length of the sequence.
    # Since we cannot use 'while' or 'for' loops, we use a recursive-like 
    # structure via a list comprehension or map, but that is tricky.
    # Alternatively, we can use the fact that for a fixed i and d, 
    # the indices are i, i+d, i+2d... 
    # We can use a list comprehension to extract the heights at these indices
    # and then use itertools.takewhile to count the prefix of identical heights.
    
    from itertools import takewhile
    
    # For every starting index i and every interval d:
    # 1. Create a sequence of heights: h[i], h[i+d], h[i+2d]...
    # 2. Use takewhile to keep elements equal to h[i].
    # 3. The length of that sequence is a candidate for the maximum.
    
    # We use a generator expression inside max()
    # range(n) for i, range(1, n) for d.
    # Note: if n=1, range(1, 1) is empty, so we handle that with a default.
    
    ans = max(
        (
            len(list(takewhile(lambda x: x == h[i], 
                               [h[i + k * d] for k in range((n - 1 - i) // d + 1)])))
            for i in range(n)
            for d in range(1, n)
        ),
        default=1
    )
    
    print(max(ans, 1))

if __name__ == "__main__":
    solve()