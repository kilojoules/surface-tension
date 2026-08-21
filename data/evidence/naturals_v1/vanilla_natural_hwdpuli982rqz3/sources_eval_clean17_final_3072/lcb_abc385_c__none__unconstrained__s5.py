import sys
from collections import defaultdict

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, H is the rest
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_map[h] = [list of indices i where H[i] == h]
    height_map = defaultdict(list)
    for i in range(N):
        height_map[H[i]].append(i)
    
    # For each height, we check all pairs of indices (i, j)
    # The distance between them is d = j - i.
    # We check how many buildings of the same height exist at intervals of d.
    # To avoid O(N^3), we can iterate through all possible starting points and intervals.
    # However, given N=3000, O(N^2) is required.
    
    # We can iterate through every pair of buildings of the same height.
    # Let the indices be idx[k] and idx[m]. The interval is d = idx[m] - idx[k].
    # The number of buildings in this sequence is (idx[m] - idx[k]) // d + 1? 
    # No, we need to verify they all have the same height.
    
    # Correct O(N^2) approach:
    # For every possible interval d (1 to N//2), and every starting position i (0 to d-1),
    # we find the longest contiguous segment of buildings with the same height.
    
    # Since we can't use for/while loops for logic, we use map/list comprehensions.
    # We want to find max(count) where H[i], H[i+d], H[i+2d]... have the same height.
    
    # Let's redefine: for a fixed d and i, we have a sequence H[i], H[i+d], H[i+2d]...
    # We need the longest run of identical values in this sequence.
    
    # To handle this without explicit loops, we can use a recursive-like 
    # structure via map or process by "layers".
    
    # Actually, the most straightforward way to get O(N^2) without 'for' loops 
    # is to use a comprehension that iterates over d and i, and then 
    # processes the sequence.
    
    # To find the longest run of identical elements in a list 'seq':
    # We can use a trick with itertools.groupby or a custom reduction.
    from itertools import groupby
    
    # We generate all possible sequences for all d and i.
    # d: interval, i: start index
    # seq: [H[i], H[i+d], H[i+2d], ...]
    # max_run: max(len(list(group)) for key, group in groupby(seq))
    
    # We use a generator expression inside max() to avoid loops.
    # The range of d is 1 to N // 2. If d > N // 2, the max possible is 1 (handled by default).
    
    ans = max(
        max(
            (len(list(group)) for key, group in groupby(H[i::d])),
            default=0
        )
        for d in range(1, (N // 2) + 1)
        for i in range(d)
    ) if N > 0 else 0
    
    # The case for N=1 or no intervals found is 1.
    print(max(ans, 1))

if __name__ == "__main__":
    solve()