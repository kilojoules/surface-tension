import sys
from itertools import groupby

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_groups = {height: [indices]}
    height_groups = {k: list(g) for k, g in groupby(sorted(enumerate(h), key=lambda x: x[1]), key=lambda x: x[1])}
    # The above groupby logic is slightly flawed because sorted(enumerate(h)) sorts by index first.
    # Let's use a more robust grouping method.
    
    # Correct grouping: Create a list of (height, index) and sort by height
    indexed_heights = sorted([(h[i], i) for i in range(n)])
    
    # Group by height and extract indices
    groups = [list(g[1]) for k, g in groupby(indexed_heights, key=lambda x: x[0])]
    
    # For each height group, we want to find the maximum number of indices that form an 
    # arithmetic progression. However, the problem says "equal intervals", 
    # meaning the distance between chosen buildings must be constant.
    # If we choose indices i, i+d, i+2d... 
    # This is equivalent to checking all possible intervals d from 1 to N.
    
    # Let's redefine: for a fixed height H, and a fixed interval d,
    # we want to find the maximum number of buildings of height H spaced d apart.
    
    # To avoid loops, we can use a comprehension that iterates over:
    # 1. All unique heights present in the input
    # 2. All possible intervals d from 1 to N
    # 3. All possible starting positions i from 0 to d-1
    
    unique_heights = set(h)
    
    # We are looking for max(count) where:
    # for height ht in unique_heights:
    #   for d in range(1, n):
    #     for i in range(d):
    #       count = sum(1 for j in range(i, n, d) if h[j] == ht)
    
    # This is O(H * N * N), which is 3000^3 (too slow).
    # Wait, the constraint is N <= 3000. We need a more efficient approach.
    # For a fixed height, we only care about the indices where that height occurs.
    
    # Let's use the groups we created: groups = [[indices of height H1], [indices of height H2], ...]
    # For a specific height's indices 'idx_list', and a potential interval 'd':
    # The number of buildings is the max length of a sequence i, i+d, i+2d... 
    # that are all present in 'idx_list'.
    
    # Actually, the simplest way to think about "equal intervals" is:
    # Pick a height 'ht', a starting index 'i', and an interval 'd'.
    # The number of buildings is the number of k >= 0 such that i + k*d < N and h[i + k*d] == ht.
    
    # To optimize: we only need to check intervals d that are differences between 
    # two indices of the same height.
    
    # Let's use a different approach:
    # For every pair of indices (i, j) with h[i] == h[j], they define an interval d = j - i.
    # But that's still O(N^2) pairs.
    
    # Let's reconsider the constraints. N=3000, O(N^2) is acceptable.
    # For every starting index i and every interval d:
    # we can't just loop. But we can check if h[i] == h[i+d] == h[i+2d]...
    
    # Correct O(N^2) approach:
    # For every possible interval d from 1 to N-1:
    #   We can group indices by (index % d).
    #   For each group, we look for the longest contiguous block of the same height.
    #   Wait, the buildings must be at EQUAL intervals. 
    #   That means if we pick interval d, we check indices i, i+d, i+2d...
    #   and count how many have the same height.
    
    # Let's use:
    # max(
    #   sum(1 for k in range(i, n, d) if h[k] == h[i])
    #   for i in range(n)
    #   for d in range(1, n)
    #   if i + d < n
    # )
    # This is still O(N^3) in the worst case.
    
    # Let's refine:
    # For a fixed height 'ht', let its indices be S = {idx1, idx2, ...}.
    # We want to find max |{i, i+d, i+2d, ...} ∩ S|.
    # This is equivalent to: for every pair idx_a, idx_b in S, 
    # let d = (idx_b - idx_a) / k for some integer k.
    
    # Actually, the most straightforward O(N^2) is:
    # For every pair of indices (i, j) with h[i] == h[j]:
    #   d = j - i
    #   count = 2 + (check i-d, i-2d... and j+d, j+2d...)
    # But we can't use loops.
    
    # Let's use the property: for a fixed height 'ht' and interval 'd',
    # we want to find max_{i \in [0, d-1]} sum(1 for k in range(i, n, d) if h[k] == ht).
    
    # Total complexity: sum_{ht} sum_{d=1 to N} d * (N/d) = sum_{ht} N * N = O(H * N^2). Still too slow.
    # Wait, the sum is actually: sum_{ht} sum_{d=1 to N} N = O(H * N^2).
    
    # Let's try: for every pair (i, j) such that h[i] == h[j], 
    # they could be the first and second elements of the sequence.
    # Then d = j - i. The number of elements is 1 + (number of k such that h[j + k*d] == h[i]).
    
    # Since we cannot use loops, we can use a recursive-like structure or 
    # a comprehension that leverages the fact that we only need to check 
    # d = j - i for all i < j where h[i] == h[j].
    
    # To keep it O(N^2), we can iterate over all i and d:
    # result = max(
    #    1,
    #    max(
    #        sum(1 for k in range(i, n, d) if h[k] == h[i])
    #        for i in range(n)
    #        for d in range(1, (n - 1) // 1 + 1) # This is still O(N^3)
    #        if i + d < n
    #    )
    # )
    
    # Let's use the "group by height" and for each height, check all pairs.
    # For a fixed height, if there are M buildings, there are M^2 pairs.
    # Total pairs across all heights is at most N^2.
    # For each pair (i, j), d = j - i. We count how many k satisfy h[i + k*d] == h[i].
    # This is still O(N^3) if we are not careful.
    
    # But we only need to check d = j - i where i is the FIRST building of the sequence.
    # And we only need to check d such that i + d < n.
    
    # Final attempt at logic:
    # For every i in 0...N-1:
    #   For every d in 1...N-i-1:
    #     if h[i] == h[i+d]:
    #       count = 2 + sum(1 for k in range(i + 2*d, n, d) if h[k] == h[i])
    # This is O(N^3) worst case (e.g., all heights same), but the sum(1...) 
    # is only called when h[i] == h[i+d].
    # For the "all heights same" case, it's N * N * (N/d), which is N^2 log N.
    # For the "all heights different" case, it's N^2.
    # This should pass.
    
    print(max(
        [
            1
        ] + [
            2 + sum(1 for k in range(i + 2*d, n, d) if h[k] == h[i])
            for i in range(n)
            for d in range(1, n - i)
            if h[i] == h[i + d]
        ]
    ))

if __name__ == "__main__":
    solve()