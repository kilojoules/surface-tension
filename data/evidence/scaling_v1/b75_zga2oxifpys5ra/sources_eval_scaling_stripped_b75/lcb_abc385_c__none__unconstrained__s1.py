import sys
from itertools import groupby

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # Group indices by their building height
    # height_groups = {height: [index1, index2, ...]}
    height_groups = {}
    for idx, height in enumerate(h):
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(idx)

    # For each height group, we need to find the maximum number of indices
    # that form an arithmetic progression.
    # Since we need equal intervals, we can iterate through all pairs of indices
    # (i, j) in a group, treat them as the first two elements of the sequence,
    # and calculate how many subsequent elements fit the pattern.
    
    # However, a more efficient way to check all equal intervals for a specific height:
    # For a fixed height and a fixed interval 'd', we count occurrences.
    # But the number of heights is up to 3000 and N is 3000.
    # The most straightforward approach is to iterate through all pairs in each group.
    
    # To optimize: for each height group, try all possible intervals d from 1 to N.
    # For a fixed height and interval d, the maximum number of buildings is
    # max(count of indices in group that belong to the same congruence class modulo d).
    
    # Let's refine: 
    # For each height 'val', let S be the set of indices.
    # We want to find max |{i, i+d, i+2d, ...}| such that all are in S.
    
    # Since we cannot use loops, we use a comprehension that iterates over 
    # all possible starting points and intervals.
    # But that's O(N^3). With N=3000, we need something faster.
    # Wait, the condition is "equal intervals", meaning indices are i, i+d, i+2d...
    # For a fixed height and fixed d, we can group indices by (index % d).
    # The size of the largest group is the answer for that d.
    
    # To avoid O(N^2 * unique_heights), we only iterate over d that are 
    # differences between indices of the same height.
    
    # Actually, the simplest O(N^2) approach:
    # For every pair of indices (i, j) with the same height, they define an interval d = j - i.
    # We can check how many k = j + d, j + 2d... also have that height.
    # But that's still potentially slow.
    
    # Let's use the property: for a fixed height and interval d, 
    # we are looking for the longest sequence.
    # For a fixed height, we can iterate through all indices i and all possible intervals d.
    # But we only care about d such that i+d is also an index of that height.
    
    # Correct O(N^2) approach:
    # For each height group, for each pair of indices (i, j) in the group:
    # The interval is d = j - i. The number of elements is 1 + (count of k in group 
    # such that k = i + m*d).
    # This is still slow. 
    
    # Let's reconsider: for a fixed height and fixed d, 
    # we group indices by (idx % d) and count.
    # But we only need to check d from 1 to N.
    # Total complexity: sum_{heights} (N * len(group)) = N * N = 3000^2 = 9,000,000.
    # This fits in the time limit.
    
    # We use a generator expression inside max() to find the result.
    # For each height group, for each d in 1..N, group indices by (idx % d).
    
    # To make it fully Pythonic and avoid loops:
    # We can use a list comprehension to iterate over height groups, 
    # and inside, another to iterate over d, and inside that, groupby.
    
    # Optimization: we only need to check d from 1 to N // (current_max_found).
    # Since we can't use loops to update current_max, we just check all d.
    
    # Using a helper logic to count:
    # For a fixed height group 'g' and interval 'd', 
    # the number of elements is max(len(list(k)) for k in groupby(sorted([i % d for i in g])))
    # Wait, that's not correct. i % d only tells us they are in the same "slot", 
    # not that they are spaced exactly by d.
    # The condition "equal intervals" means indices are i, i+d, i+2d...
    # This means they must be in the same congruence class AND there must be no gaps.
    # Actually, the problem says "chosen buildings are arranged at equal intervals".
    # This means if we choose indices p_1 < p_2 < ... < p_k, then p_2 - p_1 = p_3 - p_2 = ... = d.
    # This is exactly an arithmetic progression.
    
    # For a fixed height and fixed d, we can use a dictionary/array to count 
    # contiguous blocks. But since we can't use loops, we can use the fact that
    # for a fixed height and fixed d, we are looking for the longest run of 
    # indices in the set S that differ by d.
    
    # Let's use the O(N^2) approach:
    # For each height group, and for each pair of indices (i, j) in the group,
    # they could be the first and second elements of the sequence.
    # The length would be 1 + (number of k in group such that k = j + m*(j-i)).
    # This is still O(N^3) in worst case.
    
    # Wait, the constraints are N=3000. O(N^2) is required.
    # For a fixed height and fixed d, we can check all indices in O(len(group)).
    # Total complexity: sum_{heights} (N * len(group)) = N^2.
    # How to count the longest run without a loop?
    # For a fixed height group S and interval d:
    # We can create a boolean array (or set) of indices.
    # Then we can use a recursive-like structure or a clever comprehension.
    # Actually, we can just check all i in S and all d from 1 to N.
    # For a fixed i and d, the length is the number of k >= 0 such that (i + k*d) is in S.
    
    # To do this without a loop:
    # length = sum(1 for k in range(0, (n-1)//d + 1) if (i + k*d) in s_set)
    
    # Total complexity: sum_{heights} sum_{i in group} sum_{d=1 to N} 1
    # This is O(N^3). We need to optimize.
    # We only need to check d = (j - i) for j in group, j > i.
    # Total complexity: sum_{heights} sum_{i, j in group} (N // (j-i))
    # This is roughly O(N^2 log N).
    
    # Let's implement this using nested comprehensions.
    
    # We use a set for O(1) lookup.
    # For each height, we get the indices.
    # For each pair of indices i, j (i < j), we calculate d = j - i.
    # Then we count how many k = 0, 1, ... satisfy (i + k*d) in s_set.
    # Note: we only need to check pairs (i, j) where i is the FIRST element.
    # To ensure i is the first, we check that (i - d) is not in s_set.
    
    # Final logic:
    # 1. Group indices by height.
    # 2. For each group S:
    #    a. Create a set s_set from S.
    #    b. For each pair i, j in S (i < j):
    #       d = j - i
    #       if (i - d) not in s_set:
    #           count = sum(1 for k in range(0, (n-1)//d + 1) if (i + k*d) in s_set)
    #           update max.
    
    # To avoid the 'if' and 'update', we use a comprehension and max().
    # We handle the case where only 1 building is chosen by initializing max with 1.
    
    # Since we can't use loops, we wrap the logic in a function and use comprehensions.
    
    # To avoid O(N^3), we only iterate d = j - i.
    # The number of k's to check is N // d.
    # The total complexity is sum_{i, j} N/(j-i), which is O(N^2 log N).
    
    # Implementation:
    res = max([
        max([
            sum(1 for k in range(0, (n - 1) // (j - i) + 1) 
                if (i + k * (j - i)) in s_set)
            for i in group for j in group if i < j and (i - (j - i)) not in s_set
        ] + [1])
        for group, s_set in [(g, set(g)) for g in height_groups.values()]
    ])
    
    print(res)

if __name__ == "__main__":
    solve()