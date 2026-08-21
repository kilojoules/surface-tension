import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # If there are no buildings, the answer is 0
    if N == 0:
        print(0)
        return

    # We need to find the maximum k such that there exists a height 'h',
    # a starting index 'i', and an interval 'd' where:
    # H[i] == H[i + d] == H[i + 2d] == ... == H[i + (k-1)d] == h
    # and i + (k-1)d < N.
    
    # Since N is small (3000), we can iterate through all possible 
    # starting pairs (i, j) which define the height h and the interval d.
    
    # To avoid nested loops with if/else, we can use a approach where we 
    # check every possible interval d from 1 to N//2.
    # For a fixed d, we can check the sequences.
    
    # However, a more straightforward way to conceptualize this without 
    # complex loops is to realize that for any two buildings of the same height
    # at indices i and j (i < j), they form an interval d = j - i.
    # We can then check how many subsequent buildings at that interval have the same height.
    
    # To satisfy constraints and avoid "for" loops for the counting part,
    # we can use a technique with list comprehensions or map, but the 
    # most efficient way to handle "equal intervals" is to iterate 
    # through all possible intervals d and all possible starts i.
    
    # Given the constraints (N=3000), an O(N^2) solution is required.
    # We can iterate through all possible intervals d (1 to N//2).
    # For each d, we can partition the buildings into d groups based on (i % d).
    # In each group, we look for the longest contiguous segment of identical heights.
    
    # To avoid explicit Python loops for the "contiguous segment" part, 
    # we can use a trick with itertools.groupby or a custom reduction.
    
    from itertools import groupby
    from operator import itemgetter
    
    # We generate all possible intervals d.
    # For each d, we create strings or tuples of heights for each offset r in 0...d-1.
    # Then we find the max length of consecutive identical elements.
    
    # To truly avoid "for" loops for the logic, we use map and list comprehensions.
    
    # Function to find the max consecutive identical elements in a list
    def max_consecutive(lst):
        if not lst: return 0
        # groupby returns groups of identical consecutive elements
        # we take the length of each group and find the max
        return max([len(list(g)) for k, g in groupby(lst)])

    # We iterate d from 1 to N//2. For each d, we check all offsets r.
    # The case d=0 is handled by the fact that we can always pick 1 building.
    
    # We use a list comprehension to iterate through all d and r.
    # The inner part calculates the max consecutive for that specific (d, r).
    
    results = [
        max_consecutive([H[i] for i in range(r, N, d)])
        for d in range(1, (N // 2) + 1)
        for r in range(d)
    ]
    
    # The answer is the max of these results, or 1 if N > 0.
    print(max(results) if results else 1)

if __name__ == "__main__":
    solve()