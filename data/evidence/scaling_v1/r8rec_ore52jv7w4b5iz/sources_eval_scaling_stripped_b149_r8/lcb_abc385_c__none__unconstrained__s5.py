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
    # height_groups = {height: [indices]}
    height_groups = {k: [i for i, val in enumerate(h) if val == k] 
                     for k in set(h)}
    
    # For each height group, we need to find the maximum number of indices
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices
    # in each group to define a starting point and a common difference.
    
    # We use a comprehension to find the max length for each height group.
    # For a group of indices 'indices', we check every pair (i, j) as the 
    # first two elements of the sequence.
    # The common difference is d = indices[j] - indices[i].
    # We then count how many elements in the sorted 'indices' fit the pattern.
    
    # To avoid loops, we use a nested comprehension.
    # However, a simpler approach for "equal intervals" is to check all possible
    # intervals 'd' from 1 to N and all possible start positions 's'.
    
    # Let's use the interval-based approach:
    # For each unique height 'ht', and each possible interval 'd',
    # and each start position 's', count how many buildings have height 'ht'.
    
    unique_heights = set(h)
    
    # We want to find max(count) where:
    # for ht in unique_heights:
    #   for d in range(1, n):
    #     for s in range(d):
    #       count = sum(1 for i in range(s, n, d) if h[i] == ht)
    
    # This is O(U * N * N) where U is unique heights. That might be too slow.
    # Better: For each height, get its indices. For every pair of indices, 
    # calculate the interval and count.
    
    # Let's refine: for each height, we have a list of indices.
    # For every pair of indices (idx1, idx2) in that list, they define an interval d.
    # We can then check how many other indices in that list fit the sequence.
    
    # Actually, the most efficient way given the constraints is:
    # For every possible interval d (1 to N), and every starting position s (0 to d-1),
    # the buildings at s, s+d, s+2d... form a sequence.
    # In that sequence, we just need to find the most frequent height.
    
    # result = max(
    #     count of most frequent height in sequence (s, s+d, s+2d...)
    #     for d in range(1, n + 1)
    #     for s in range(d)
    # )
    
    # To implement "count of most frequent height" without loops:
    # We can use groupby on the sequence sorted by height.
    
    ans = max(
        max(
            [len(list(g)) for k, g in groupby(sorted([h[i] for i in range(s, n, d)]))]
        )
        for d in range(1, n + 1)
        for s in range(d)
    )
    
    print(ans)

# Using a slightly different structure to fit the "no loop" constraint strictly
# and ensuring it handles the N=1 case.
def final_solve():
    input_data = sys.stdin.read().split()
    if not data := input_data: return
    n = int(data[0])
    h = list(map(int, data[1:]))
    
    # For every possible interval d and start s, 
    # extract the heights and find the max frequency of any height.
    # We use a generator expression inside max().
    print(max(
        max(
            [len(list(g)) for k, g in groupby(sorted([h[i] for i in range(s, n, d)]))]
        )
        for d in range(1, n + 1)
        for s in range(d)
    ))

if __name__ == "__main__":
    final_solve()