```python
import sys
from itertools import groupby

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups: {height: [index1, index2, ...]}
    height_groups = {}
    for i, h in enumerate(H):
        if h not in height_groups:
            height_groups[h] = []
        height_groups[h].append(i)
    
    # For each height, we check all possible intervals 'd'
    # An interval 'd' is valid if we can pick indices i, i+d, i+2d...
    # The maximum number of buildings for a specific height and interval d
    # is the maximum length of a sequence of indices in that height's index list
    # that forms an arithmetic progression with common difference d.
    
    # We use a comprehension to iterate through all heights and all possible intervals.
    # For a fixed height and interval d, we check how many indices k satisfy:
    # (index % d) is the same, and we count the occurrences.
    # However, the condition is "equal intervals" in the original line, 
    # meaning indices must be i, i+d, i+2d... 
    # This means for a fixed d, we group indices by (index % d).
    # The number of buildings is the count of indices in that group, 
    # BUT they must actually be spaced by exactly d.
    # Wait, the condition "arranged at equal intervals" means if we pick 
    # indices p1, p2, ..., pk, then p2-p1 = p3-p2 = ... = pk-p_{k-1} = d.
    # This is stricter than just (index % d) being constant.
    # It means we are looking for the longest arithmetic progression in the 
    # set of indices for each height.
    
    # Since N is small (3000), we can iterate through all pairs of indices 
    # for each height to define a starting point and a difference.
    # But a simpler way: for every height, and every possible difference d (1 to N),
    # and every starting index s (0 to d-1), check how many indices in the 
    # height's index set match the pattern s, s+d, s+2d...
    
    # Let's refine: For a fixed height and difference d, we want to find
    # the maximum number of indices i such that i = s + k*d.
    # This is equivalent to counting how many i in the index set satisfy i % d == s,
    # PROVIDED that the indices are contiguous in the sequence s, s+d, s+2d...
    # Actually, the problem says "arranged at equal intervals", which implies
    # the indices are i, i+d, i+2d, ... i+(k-1)d.
    # This means for a fixed height and difference d, we are looking for the 
    # longest run of indices in the set {s, s+d, s+2d, ...} that are all present.
    
    # Correct approach:
    # For each height's indices set S:
    #   For each d from 1 to N:
    #     For each s from 0 to d-1:
    #       Find the longest sequence of k such that (s + k*d) is in S.
    
    # To optimize, we can use the fact that we only care about d that are 
    # differences between existing indices of the same height.
    
    # Let's use a different approach: 
    # For each height, get the sorted list of indices.
    # For every pair of indices (i, j) in that list, they define a difference d = j - i.
    # We can then check how many terms of the progression i, i+d, i+2d... are in the set.
    
    # Given N=3000, O(N^2) is acceptable.
    # We can iterate through all pairs of indices (i, j) and for each, 
    # calculate the length of the progression. But that's O(N^3) worst case.
    # Instead, we can use DP: dp[i][d] is the length of the progression ending at i with diff d.
    # But d can be up to 3000. DP table size N*N is 9*10^6, which is fine.
    
    # Let's use the property: for a fixed height, we only care about its indices.
    # For each height group:
    #   indices = sorted(height_groups[h])
    #   n_h = len(indices)
    #   dp = [{} for _ in range(n_h)]
    #   for j in range(n_h):
    #     for i in range(j):
    #       d = indices[j] - indices[i]
    #       dp[j][d] = dp[i].get(d, 1) + 1
    #   result = max(dp[j].values())
    
    # To avoid loops, we can use a generator and map/reduce, but the DP is inherently sequential.
    # However, we can implement the DP using a loop and then wrap it in a function.
    
    def get_max_for_height(indices):
        n_h = len(indices)
        if n_h == 0: return 0
        # dp[j] will store {difference: length}
        dp = [{} for _ in range(n_h)]
        # We must use a loop here as DP is sequential
        for j in range(n_h):
            for i in range(j):
                d = indices[j] - indices[i]
                # If the previous index was exactly indices[i], 
                # and the one before that was indices[i] - d, 
                # then dp[i][d] would have been tracked.
                # But we need to ensure the interval is constant.
                # The condition is: indices are i, i+d, i+2d...
                # So we check if indices[i] - d was also in the set.
                # Actually, the simplest DP is:
                # dp[j][d] = dp[i][d] + 1 if indices[j] - indices[i] == d
                # Since we iterate i from 0 to j-1, we can just do:
                dp[j][d] = dp[i].get(d, 1) + 1
        
        # Extract all lengths and find the max
        all_lengths = [length for group in dp for length in group.values()]
        return max(all_lengths) if all_lengths else 1

    # Since I cannot use loops, I will use a recursive-like structure or 
    # a different approach. Wait, the constraint is "no for/while loops".
    # I can use a list comprehension to simulate the DP by processing 
    # the indices and updating a state. But state update in list comprehension 
    # is tricky.
    
    # Alternative: For every pair of indices (i, j) of the same height, 
    # they define a potential interval d = j - i.
    # The number of elements is (j - i) // d + 1? No, that's only if all are present.
    # The number of elements is the count of k such that (i + k*d) is in the set.
    
    # Let's use the property: for a fixed height and a fixed difference d,
    # we can group indices by (index % d). 
    # For each group, we check if the indices form a contiguous range with step d.
    # A group of indices {p1, p2, ..., pk} sorted forms a progression with step d 
    # if p_{m+1} - p_m = d for all m.
    # This means the length of the progression is the maximum number of 
    # consecutive integers in the set {p // d}.
    
    # For a fixed height and difference d:
    # 1. Filter indices: S = {i for i in height_indices}
    # 2. For each s in 0..d-1:
    #    Consider the set T = { (i-s)//d for i in S if i % d == s }
    #    Find the longest run of consecutive integers in T.
    
    # To avoid loops, we can use a helper function and map.
    def count_consecutive(sorted_ints):
        # Group by (value - index) to find consecutive sequences
        # Example: [1, 2, 3, 5, 6] -> (1-0)=1, (2-1)=1, (3-2)=1, (5-3)=2, (6-4)=2
        # Groups: [1, 1, 1] and [2, 2]
        groups = [list(g) for k, g in groupby([v - i for i, v in enumerate(sorted_ints)])]
        return max(map(len, groups)) if groups else 0

    # We need to check all d from 1 to N.
    # For each height, and each d, and each s... this is too many.
    # Actually, we only need to check d that are differences between some indices of that height.
    
    # Let's use a different approach:
    # For each height, and for each pair of indices (i, j) with i < j:
    # d = j - i. The number of elements is 1 + (count of k > 0 such that j + k*d is in S).
    # This is still O(N^3).
    
    # But we can use the "consecutive integers" trick for all d in 1..N.
    # For a fixed height and fixed d:
    # The max length is max(count_consecutive(sorted([i // d for i in S if i % d == s])))
    # for all s in 0..d-1.
    # This is equivalent to:
    # For a fixed height and fixed d, group indices i in S by (i % d).
    # For each group, sort the values i // d and find the longest consecutive run.
    
    # To avoid loops, we use map and list comprehensions.
    
    # We can't iterate d from 1 to N using a loop. We use range(1, N).
    # We can't iterate heights using a loop. We use height_groups.keys().
    
    # The total complexity would be O(H * N * (N/d)) which is O(H * N log N).
    # With H=3000, N=300s, this is too slow. 
    # Wait, the number of heights is at most N.
    # The total number of indices across all heights is N.
    # For a fixed d, we process each index once. Total O(N * N).
    # 3000 * 3000 = 9 * 10^6. This should pass in Python if implemented efficiently.

    # Final logic:
    # For each d in 1..N:
    #   For each height h:
    #     Indices S_h = height_groups[h]
    #     For each s in 0..d-1:
    #       T = sorted([i // d for i in S_h if i % d == s])
    #       ans = max(ans, count_consecutive(T))
    
    # To optimize: instead of s in 0..d-1, just group S_h by (i % d).
    
    def solve_for_d(d):
        # For a given d, find the max length across all heights
        # We group indices of each height by (i % d)
        # Then for each group, we find the longest consecutive run of i // d
        return max(
            (
                max(
                    (
                        count_consecutive(sorted([i // d for i in group]))
                        for _, group in groupby(
                            sorted([i for i in height_groups[h]]), 
                            key=lambda x: x % d
                        )
                    ),
                    default=0
                )
                for h in height_groups
            ),
            default=0
        )

    # The above is still a bit slow. Let's refine.
    # For a fixed d, we can process all heights together.
    # For each index i, it belongs to height H[i].
    # We are interested in i, i+d, i+2d... having the same height.
    # This is a run of the same height in the sequence:
    # For a fixed start s in 0..d-1:
    # Sequence: H[s], H[s+d], H[s+2d], ...
    