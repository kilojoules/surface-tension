import sys
from itertools import groupby

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups: {height: [index1, index2, ...]}
    # We use a dictionary comprehension to organize indices of the same height
    # Since we cannot use loops, we use a list comprehension to build a list of tuples
    # and then a dictionary to group them. However, a simpler way is to 
    # iterate through all unique heights and filter the indices.
    
    unique_heights = set(h)
    
    # For each height, we find all indices where that height occurs.
    # Then, for every pair of indices (i, j) in that set, we calculate the interval d = j - i.
    # We check how many elements in the sequence (i, i+d, i+2d...) have the same height.
    
    # To avoid loops, we use nested list comprehensions.
    # 1. Get indices for each height: [[idx for idx, val in enumerate(h) if val == ht] for ht in unique_heights]
    # 2. For each set of indices, try all possible intervals d.
    # 3. For a fixed height, start index i, and interval d, count how many k satisfy h[i + k*d] == height.
    
    # Optimization: Instead of checking all d, we only check d that are differences between existing indices.
    # But since N=3000, a O(N^2) approach is needed. 
    # We can iterate through all pairs (i, j) and check the sequence.
    
    # Let's refine: for every pair (i, j) with h[i] == h[j], they define an interval d = j - i.
    # The number of buildings is (j - i) // gcd(j-i, ...) ? No.
    # The simplest O(N^2) is: for every pair (i, j), if h[i] == h[j], 
    # the count is 1 + (number of k > 0 such that i + k*(j-i) < N and h[i + k*(j-i)] == h[i]).
    
    # To implement this without loops:
    # We use a list comprehension to evaluate all pairs (i, j) where i < j and h[i] == h[j].
    # For each pair, we use another list comprehension to count the valid k's.
    
    res = [
        1 + sum(1 for k in range(1, (n - 1 - i) // (j - i) + 1) 
                if h[i + k * (j - i)] == h[i])
        for i in range(n)
        for j in range(i + 1, n)
        if h[i] == h[j]
    ]
    
    # The answer is the maximum of res, or 1 if res is empty.
    print(max(res) if res else 1)

if __name__ == "__main__":
    solve()