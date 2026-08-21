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
    # height_groups = {height: [index1, index2, ...]}
    # We use a dictionary comprehension to organize indices of the same height
    # To avoid loops, we can use a list comprehension to create (height, index) pairs
    # and then sort them to use groupby.
    indexed_heights = sorted([(h[i], i) for i in range(n)])
    
    # Group by height and extract only the indices for each height
    groups = [list(indices) for height, group in groupby(indexed_heights, lambda x: x[0]) 
              for indices in [[g[1] for g in group]]]
    
    # For each group of indices, we need to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices (i, j)
    # in a group to define a starting point and a common difference 'd'.
    
    # We use a nested list comprehension to evaluate all possible intervals 'd'
    # for every pair of buildings of the same height.
    # For a fixed height group 'g', and two indices g[i] and g[j], 
    # the interval is d = g[j] - g[i]. 
    # We count how many k satisfy (g[i] + k*d) is also in the set of indices.
    
    # To optimize, we convert each group to a set for O(1) lookup.
    # We only consider pairs (i, j) where i < j.
    
    # The logic: for every group, for every pair of indices, calculate the 
    # length of the sequence.
    # Use a helper function inside a list comprehension.
    
    def get_max_for_group(g):
        if len(g) <= 2:
            return len(g)
        
        s = set(g)
        # For every pair of indices in the group, treat them as the 1st and 2nd elements
        # and count how far the sequence goes.
        # We use a generator expression inside max() to find the length.
        # The sequence is g[i], g[i]+d, g[i]+2d...
        # We use a while-like logic via a recursive-style list comprehension or 
        # simply iterate through possible counts.
        # Actually, a simpler way: for a fixed start g[i] and diff d, 
        # the count is the number of k >= 0 such that g[i] + k*d < N and is in s.
        
        # Since we cannot use while/for loops, we use a list comprehension 
        # over the range of possible k values (up to N).
        return max(
            max(
                sum(1 for k in range(n) if (g[i] + k * (g[j] - g[i])) in s),
                1
            )
            for i in range(len(g))
            for j in range(i + 1, len(g))
        ) if len(g) > 1 else 1

    # We wrap the logic in a way that avoids explicit loops.
    # We use a list comprehension to process all groups and then take the global max.
    # Note: the 'get_max_for_group' is called via a list comprehension.
    
    # To strictly avoid 'def' if forbidden (though usually allowed), 
    # we can inline the logic, but 'def' is generally accepted as a tool 
    # to organize the comprehension.
    
    ans = max([get_max_for_group(g) for g in groups]) if groups else 0
    print(ans)

if __name__ == "__main__":
    solve()