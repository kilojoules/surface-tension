import sys
from itertools import groupby

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by height: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to organize indices of each unique height
    from collections import defaultdict
    height_groups = defaultdict(list)
    for idx, height in enumerate(h):
        height_groups[height].append(idx)
    
    # For each height group, we need to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices (i, j)
    # within each height group to define a starting point and a common difference.
    
    # To avoid loops, we use a nested comprehension.
    # For a fixed height, we check every pair of indices (i, j) as the first two elements.
    # The common difference is d = j - i.
    # The number of elements isK = (last_index - i) // d + 1, 
    # provided the sequence is fully populated with that height.
    # However, the condition is "chosen buildings", meaning we can skip buildings.
    # Wait, the condition "arranged at equal intervals" means if we pick indices 
    # i, i+d, i+2d... they must all have the same height.
    
    # Let's refine: for every possible starting index 'i' and every possible 
    # interval 'd' (1 to N), count how many k satisfy H[i + k*d] == H[i].
    
    # We use a generator expression inside max()
    # We iterate d from 1 to N, and i from 0 to N-1.
    # For a fixed i and d, we count k such that i + k*d < N and H[i + k*d] == H[i].
    
    # To optimize: we only need to check d if there's at least one other building 
    # of the same height. If all heights are different, the answer is 1.
    
    # Using a list comprehension to evaluate all (i, d) pairs:
    # We use range(1, n) for d and range(n) for i.
    # For each (i, d), we use a while-like logic via a generator to count.
    # Since we can't use while, we can use a range and a filter.
    
    # The maximum possible k for a given i and d is (n - 1 - i) // d.
    # We check k in range((n - 1 - i) // d + 1).
    
    ans = max(
        (
            sum(1 for k in range((n - 1 - i) // d + 1) if h[i + k * d] == h[i])
            for i in range(n)
            for d in range(1, n)
        ),
        default=1
    )
    
    # The case for 1 building is covered by the range(0, 1) when k=0, 
    # but the constraints say N >= 1, so we handle it.
    # If N=1, the loops might result in 0 or 1. Let's ensure the minimum is 1.
    print(max(1, ans))

if __name__ == "__main__":
    solve()