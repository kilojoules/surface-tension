import sys
from itertools import groupby

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings with the same height H_i
    # that are spaced at equal intervals 'd'.
    # For a fixed height 'h' and interval 'd', we check how many buildings 
    # of height 'h' exist at indices i, i+d, i+2d...
    # However, it is simpler to iterate through all possible intervals d (1 to N)
    # and all possible starting positions i (0 to d-1).
    # For each (d, i), we look at the sequence H[i], H[i+d], H[i+2d]...
    # and find the most frequent height in that sequence.
    
    # We use a generator expression to evaluate all possible intervals and offsets.
    # d: the interval between buildings
    # i: the starting index
    # For each (d, i), we extract the sequence and find the max count of identical heights.
    
    # To avoid loops, we use map/max/groupby.
    # We iterate d from 1 to N. For each d, we iterate i from 0 to d-1.
    
    # Note: If N=1, the range(1, N) is empty, so we handle N=1 explicitly or 
    # ensure the range covers the base case.
    
    # For a fixed d and i, the sequence is H[i::d].
    # To find the max frequency of a height in H[i::d]:
    # 1. Sort the sequence.
    # 2. Group identical elements.
    # 3. Find the length of the longest group.
    
    # Since we cannot use 'for' loops, we use nested comprehensions.
    # We use a helper function logic inside a comprehension.
    
    # The maximum possible answer is at least 1 (given N >= 1).
    # We check d from 1 up to N.
    
    ans = max(
        (
            max(
                (
                    len(list(g)) 
                    for k, g in groupby(sorted(H[i::d]))
                ), 
                default=0
            )
            for d in range(1, N + 1)
            for i in range(min(d, N))
        ), 
        default=1
    )
    
    print(ans)

if __name__ == "__main__":
    solve()