import sys
from itertools import combinations

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # If there is only one building, the answer is always 1
    if n == 1:
        print(1)
        return

    # We want to find the maximum number of buildings with the same height H
    # spaced at a constant interval D.
    # Any valid set of 2 or more buildings is defined by the first building (i),
    # the second building (j), and their height H = h[i].
    # The interval is D = j - i.
    
    # We use a list comprehension to evaluate all pairs (i, j) as potential 
    # starting points and intervals.
    # For each pair, we check all indices k = i + m*D and count those where h[k] == h[i].
    
    # To optimize, we only consider pairs (i, j) where h[i] == h[j].
    # We use combinations to pick indices i and j.
    
    results = [
        sum(1 for k in range(i, n, j - i) if h[k] == h[i])
        for i, j in combinations(range(n), 2)
        if h[i] == h[j]
    ]
    
    # The answer is the maximum of the results, or 1 if no pairs were found.
    print(max(results) if results else 1)

if __name__ == "__main__":
    solve()