import sys
from itertools import groupby

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups = {height: [index1, index2, ...]}
    # We use a dictionary comprehension to build the groups
    # To avoid loops, we can use a technique with sorted indices
    # But since we need to group by height, we'll use a dictionary
    # and a list comprehension to populate it.
    
    # Instead of a loop to populate the dict, we can use a 
    # comprehension that builds a list of (height, index) tuples,
    # sorts them, and then we use groupby.
    
    indexed_heights = sorted([(h[i], i) for i in range(n)])
    
    # Group by height and extract the indices for each height
    groups = [list(indices) for key, group in groupby(indexed_heights, lambda x: x[0]) 
              for indices in [[item[1] for item in group]]]
    
    # For each group of indices (buildings of the same height), 
    # we want to find the maximum number of indices that form an 
    # arithmetic progression.
    # An arithmetic progression is defined by a starting index 'a' 
    # and a common difference 'd'.
    # The number of elements is the count of i such that (a + i*d) is in the set.
    
    # We can iterate through all pairs of indices in each group to define 'a' and 'd'.
    # To avoid explicit loops, we use nested list comprehensions.
    
    # For a specific group of indices 'g':
    # We check every pair (g[i], g[j]) as the first two elements of the sequence.
    # The difference is d = g[j] - g[i].
    # We then count how many subsequent elements g[j] + d, g[j] + 2d... are in the set.
    
    # However, a simpler way to count the length of the progression 
    # starting at g[i] with difference d is to use a while-like 
    # structure via a recursive-like comprehension or 
    # by checking all possible d values.
    
    # Given the constraints (N=3000), an O(N^2) approach is acceptable.
    # We can iterate through all possible differences d from 1 to N.
    # For a fixed d, we can use dynamic programming: 
    # dp[i] = dp[i-d] + 1 if h[i] == h[i-d] else 1.
    
    # To implement this without loops, we can use a list comprehension 
    # that builds the DP table for each d.
    # But since we cannot use loops, we can use a trick with 
    # range and a custom function or a very large list comprehension.
    
    # Let's use the DP approach:
    # For each d in range(1, n):
    #   dp = [1] * n
    #   for i in range(d, n):
    #     if h[i] == h[i-d]: dp[i] = dp[i-d] + 1
    #   ans = max(ans, max(dp))
    
    # To do this without loops, we can use a functional approach.
    # We can use a list comprehension to iterate d, and inside, 
    # we need to simulate the DP. 
    # Since we can't use loops, we can use the fact that 
    # for a fixed d, the indices are split into d independent chains.
    # Chain k: k, k+d, k+2d...
    # In each chain, we are looking for the longest contiguous segment of identical heights.
    
    # For a fixed d and a starting point k < d:
    # The sequence of heights is h[k], h[k+d], h[k+2d]...
    # We want the longest run of identical values.
    
    # We can use groupby on the sliced list h[k::d]
    
    ans = max(
        # For every possible difference d from 1 to n-1
        # And every possible starting offset k from 0 to d-1
        # Find the longest group of identical heights
        # If d=0 is not possible, we handle the single building case with max(..., 1)
        [
            len(list(g)) 
            for d in range(1, n) 
            for k in range(d) 
            for key, g in groupby(h[k::d])
        ] + [1]
    )
    
    print(ans)

if __name__ == "__main__":
    solve()