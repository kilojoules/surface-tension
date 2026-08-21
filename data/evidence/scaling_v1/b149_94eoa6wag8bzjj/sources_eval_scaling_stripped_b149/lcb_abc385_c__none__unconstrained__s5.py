import sys
from collections import defaultdict

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_groups = {height: [index1, index2, ...]}
    height_groups = defaultdict(list)
    for i in range(n):
        height_groups[h[i]].append(i)
    
    # For each height, we want to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since we need to check all possible intervals 'd', 
    # we can iterate through all pairs of indices (i, j) for each height.
    
    # We use a list comprehension to find the max for each height group.
    # For a group of indices 'indices', we check every pair (i, j) as the 
    # first two elements of the sequence.
    # The interval is d = indices[j] - indices[i].
    # We then count how many subsequent elements indices[j] + k*d are also in the set.
    
    # To avoid loops, we can use a helper function logic inside a comprehension.
    # However, the constraint N=3000 allows O(N^2) if we are careful.
    # Actually, the condition is: chosen buildings are at equal intervals.
    # This means if we pick indices p, p+d, p+2d... 
    # we just need to check if H[p] == H[p+d] == H[p+2d]...
    
    # Let's redefine: for every possible starting position i and interval d:
    # count how many buildings have the same height as H[i].
    
    # We can use a generator expression inside max()
    # We iterate d from 1 to N-1, and i from 0 to N-1.
    # For a fixed i and d, we count k such that H[i] == H[i + k*d].
    
    # To optimize: we only need to check d such that i + d < N.
    # The number of elements is (k+1) where i + k*d < N and H[i] == H[i + k*d] == ...
    # This is still slightly recursive. Let's use the property that 
    # we can just check all i and d, and for each, count the sequence.
    
    # Since we cannot use while/for loops, we can use a recursive-like 
    # structure via a helper function or a clever comprehension.
    # But the simplest O(N^2) is: for every pair (i, j), they define a height H[i] 
    # and a distance d = j - i. We check how many indices i + k*d have height H[i].
    
    # Let's use the height_groups to limit the search space.
    # For each height, we check all pairs of indices.
    
    def count_sequence(indices_set, start_val, d):
        # This is tricky without loops. Let's use a different approach.
        # For a fixed height and interval d, we can partition indices into 
        # congruence classes modulo d.
        pass

    # Correct approach without loops:
    # For every possible interval d (1 to N), and every starting position i (0 to N-1):
    # We want to find the longest chain i, i+d, i+2d... with the same height.
    # This looks like DP: dp[i][d] = 1 + dp[i+d][d] if H[i] == H[i+d] else 1.
    # Since we can't use loops to fill DP, we can use a recursive function with memoization.
    
    import functools
    
    @functools.lru_cache(None)
    def get_length(i, d):
        if i + d >= n:
            return 1
        if h[i] == h[i + d]:
            return 1 + get_length(i + d, d)
        return 1

    # We need to evaluate get_length for all i in [0, n-1] and d in [1, n-1].
    # The answer is the maximum of these values.
    # We use a generator expression to trigger the computations.
    
    # Note: recursion limit might be an issue for N=3000, but d is the step.
    # The maximum depth is N/d. For d=1, depth is 3000.
    sys.setrecursionlimit(5000)
    
    ans = max(
        (get_length(i, d) for i in range(n) for d in range(1, n)) 
        if n > 1 else [1]
    )
    
    print(ans if n > 0 else 0)

if __name__ == "__main__":
    solve()