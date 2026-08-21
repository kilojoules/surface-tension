import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We need to find the maximum number of buildings that:
    # 1. Have the same height H
    # 2. Are spaced at equal intervals D
    
    # We can iterate through all unique heights present in the array.
    # For each height, we find all indices where that height occurs.
    # Then we check all possible intervals between these indices.
    
    # However, a simpler approach using comprehensions:
    # Try every starting index i, every interval d, and every height val.
    # But that's O(N^3). With N=3000, we need something more efficient.
    
    # Optimized approach:
    # For every pair of indices (i, j) with the same height, they define an interval d = j - i.
    # We can check how many buildings with that same height exist at intervals of d.
    
    # To avoid loops, we use nested comprehensions.
    # Let's group indices by height first.
    height_groups = {val: [i for i, x in enumerate(h) if x == val] for val in set(h)}
    
    # For each height group, we check all possible intervals d.
    # The number of buildings for a start index 's' and interval 'd' is:
    # count(s, d) = length of sequence s, s+d, s+2d... such that all have height 'val'
    
    # Since we cannot use loops, we use a comprehension to calculate the max for each height.
    # For a fixed height 'val' and its indices 'idxs':
    # We check every pair of indices (idxs[i], idxs[j]) to define an interval d.
    # Then we count how many k satisfy (idxs[i] + k*d) is in the set of indices.
    
    # To keep it efficient and loop-free:
    # For each height, we iterate through all possible intervals d from 1 to N.
    # For each d, we check all possible starting positions s from 0 to d-1.
    
    # Actually, the most straightforward loop-free way to implement the logic:
    # max(count) for val in unique_heights, d in 1..N, s in 0..N-1
    # such that h[s] == val and we count how many s + k*d have height val.
    
    # Given N=3000, O(N^3) is too slow. But we only need to check d and s 
    # if h[s] == val.
    
    # Let's use the property: for a fixed height 'val' and interval 'd',
    # we can group indices by (index % d).
    
    # The following logic finds the max count:
    # 1. Get all unique heights.
    # 2. For each height, get the list of indices.
    # 3. For each possible interval d (1 to N), 
    #    group indices by (index % d) and find the max group size.
    #    Wait, that's only if the indices are perfectly spaced.
    #    The condition is "equal intervals", meaning index, index+d, index+2d...
    #    This means we are looking for an arithmetic progression in the indices.
    
    # Correct logic for "equal intervals":
    # For a fixed height 'val' and interval 'd', we check sequences.
    # A sequence is s, s+d, s+2d... 
    # The number of elements is the length of the contiguous block of 1s 
    # in a bitmask or a boolean array.
    
    # To avoid loops and recursion, we can use a comprehension that:
    # For each height 'val':
    #   For each interval 'd' from 1 to N:
    #     For each start 's' from 0 to d-1:
    #       Count consecutive hits of 'val' at s, s+d, s+2d...
    
    # However, the simplest O(N^2) approach is:
    # For every pair of indices (i, j) with the same height, d = j - i.
    # But we need to count the length of the progression.
    
    # Let's use the most direct comprehension:
    # For every height 'v' in the set of heights:
    #   For every interval 'd' from 1 to N:
    #     For every starting index 's' from 0 to N-1:
    #       If h[s] == v, count how many k >= 0 satisfy s + k*d < N and h[s + k*d] == v.
    #       Crucially, the condition is "chosen buildings are arranged at equal intervals".
    #       This means if we pick indices i_1, i_2, ..., i_m, then i_{j+1} - i_j = d.
    #       This is exactly a sequence s, s+d, s+2d... 
    #       BUT, the buildings in between (s+d/2 etc) do NOT have to be the same height.
    #       They just cannot be "chosen".
    #       Wait, the condition is: "The chosen buildings all have the same height" AND "arranged at equal intervals".
    #       This means we pick indices {s, s+d, s+2d, ..., s+(m-1)d} and all must have height H.
    
    # To implement this without loops:
    # We can use a nested comprehension to evaluate all (s, d) pairs.
    # For a fixed s and d, the number of buildings is the length of the 
    # prefix of the sequence s, s+d, ... that all have the same height.
    # No, it doesn't have to be a prefix. It can be any sequence.
    # Actually, the problem says "The chosen buildings are arranged at equal intervals".
    # This implies the indices are s, s+d, s+2d, ..., s+(m-1)d.
    # All these must have the same height.
    
    # To avoid explicit loops, we use:
    # max( [count for s in range(N) for d in range(1, N) ...])
    
    # Since N=3000, O(N^3) is too slow. But we only need to check d such that 
    # there's at least one other building of the same height.
    
    # Let's refine:
    # For each height 'v', let 'indices' be the sorted list of indices where h[i] == v.
    # For every pair of indices (i, j) in 'indices', d = j - i.
    # We want to find the longest chain.
    
    # Actually, the simplest way to think about it:
    # For a fixed height 'v' and interval 'd', we can partition indices into 
    # congruence classes modulo 'd'. In each class, we look for the longest 
    # sequence of indices that form an arithmetic progression with difference 'd'.
    # Since the indices must be s, s+d, s+2d..., this means in the 
    # boolean array B where B[i] = (h[i] == v), we are looking for the 
    # longest run of 1s in the subsequence B[s], B[s+d], B[s+2d]...
    
    # But the problem says "The chosen buildings... are arranged at equal intervals".
    # This means we pick a set of indices {i_1, i_2, ..., i_m} such that 
    # i_{j+1} - i_j = d for some d > 0.
    # This is exactly a sequence s, s+d, s+2d, ..., s+(m-1)d.
    # All these must have the same height.
    
    # To avoid O(N^3), we can iterate over all pairs (i, j) that have the same height.
    # Let d = j - i. We check if this pair can be part of a longer chain.
    # However, the simplest loop-free way to write this is:
    
    # result = max(
    #    [ 
    #      sum(1 for k in range(0, (n-s-1)//d + 1) if h[s + k*d] == h[s])
    #      for s in range(n) 
    #      for d in range(1, n) 
    #      if s + d < n and h[s] == h[s+d]
    #    ] + [1]
    # )
    # This is still O(N^3) in worst case. Let's optimize.
    # We only need to check d that are divisors of (j - i).
    # Actually, the most efficient way is:
    # For each height v:
    #   For each pair of indices i, j with height v:
    #     d = j - i
    #     (This is still potentially O(N^3))
    
    # Wait, the constraint to avoid loops forces a functional style.
    # Let's use a comprehension that iterates over all pairs (i, j) 
    # and for each pair, we don't need to count the whole chain, 
    # we can use dynamic programming. But DP requires loops.
    
    # Let's reconsider the constraints. N=3000. O(N^2) is acceptable.
    # For a fixed d, we can find the longest run of the same height in O(N).
    # Total complexity O(N^2).
    
    # How to do "longest run of 1s" in a subsequence without loops?
    # For a fixed d and s, the sequence is h[s], h[s+d], h[s+2d]...
    # We can use `itertools.groupby` to find runs of the same height.
    
    import itertools
    
    # We want to find max(len(list(group))) 
    # for d in range(1, n)
    # for s in range(d)
    # for k, group in groupby([h[i] for i in range(s, n, d)])
    
    # This is O(N^2). Let's write it as a one-liner.
    
    ans = max([
        max([len(list(g)) for k, g in itertools.groupby([h[i] for i in range(s, n, d)])])
        for d in range(1, n)
        for s in range(min(d, n))
    ] + [1])
    
    print(ans)

if __name__ == "__main__":
    solve()