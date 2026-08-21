import sys
from functools import reduce

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    if n == 0:
        print(0)
        return

    # We want to find max count of buildings with same height at equal intervals.
    # An interval is defined by d (distance) and r (start offset).
    # For a fixed d and r, we look at heights: h[r], h[r+d], h[r+2d]...
    # We want the most frequent height in that sequence.
    
    # To avoid loops, we use comprehensions.
    # We iterate d from 1 to n-1. For each d, we check all possible offsets r < d.
    # For each (d, r), we extract the sequence and find the max frequency.
    
    # However, O(N^2) is 3000^2 = 9,000,000, which is acceptable in Python 
    # if we use built-ins.
    
    # We can use a helper to get the max frequency of a list.
    # Since we can't define functions, we use a trick with groupby or a counter.
    # But we can't import Counter. We can use sorted() and groupby().
    
    # Let's use a more direct approach:
    # For every pair of indices (i, j) with h[i] == h[j], they define an interval d = j - i.
    # This is still potentially O(N^3) if we check all k.
    
    # The most reliable O(N^2) is:
    # For each d in range(1, n):
    #   For each r in range(d):
    #     Sequence S = [h[i] for i in range(r, n, d)]
    #     Result = max(frequency of heights in S)
    
    # To implement "max frequency" without loops or Counter:
    # sorted(S) -> groupby(S) -> max(len(list(g)) for k, g in groupby(S))
    
    # To avoid the O(N^2 * log N) from sorting, we can't. But N=3000 is small enough.
    # Actually, the number of elements in S is n/d.
    # Total complexity: Sum_{d=1 to n} Sum_{r=0 to d-1} (n/d log n/d) 
    # = Sum_{d=1 to n} n log(n/d) approx n * n = O(N^2).
    
    # Let's refine the comprehension:
    res = max(
        max(
            (
                max(
                    [len(list(g)) for k, g in groupby(sorted([h[i] for i in range(r, n, d)]))]
                ),
                1
            )
            for r in range(d)
        )
        for d in range(1, n)
    ) if n > 1 else 1

    # Wait, the above is still a bit slow. Let's use a simpler O(N^2):
    # For every pair i, j where h[i] == h[j], they could be the 1st and 2nd elements.
    # But that's O(N^3).
    
    # Let's use the property that we only care about heights that appear multiple times.
    # For each height, get its indices. For every pair of indices, calculate d.
    # This is still O(N^3) if one height appears N times.
    
    # The most efficient way is to iterate through all possible intervals d.
    # For a fixed d, we can process the array in O(N).
    
    # Let's use a list comprehension to implement the O(N^2) logic.
    # We use a helper to get the max frequency.
    
    print(res)

# Since I cannot use 'def' inside the block for the logic, 
# I will wrap the logic in a way that it executes.
# But the prompt says "Complete Python program". 
# I will use a list comprehension that performs the O(N^2) check.

import sys
from itertools import groupby

def main():
    input_data = sys.stdin.read().split()
    if not input_data: return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    if n <= 1:
        print(n)
        return

    # We need to find max(count) where h[r] == h[r+d] == h[r+2d]...
    # Instead of all d, we only check d that are differences between indices of the same height.
    # To keep it O(N^2), we can iterate d from 1 to n.
    # For each d, we check all r < d.
    # For each (d, r), we count the most frequent height.
    
    # To optimize: we only care about d if it's a divisor of (j-i) for some h[i]==h[j].
    # Actually, the simplest O(N^2) is:
    # For each height 'val', let its indices be 'idx'.
    # For every pair i, j in 'idx', d = j - i.
    # Then count how many k in 'idx' satisfy (k - i) % d == 0.
    # This is still O(N^3).
    
    # Correct O(N^2):
    # For each height 'val', let its indices be 'idx'.
    # For every pair i, j in 'idx' (i < j):
    #   d = j - i
    #   The number of elements is (count of k in 'idx' such that k = i + m*d).
    #   This is still O(N^3).
    
    # Wait, the constraint N=3000 and 2 seconds means O(N^2) is mandatory.
    # The only way to get O(N^2) is to iterate d and r, and then 
    # count heights. But that's O(N^2) total.
    # Total elements processed: Sum_{d=1 to N} Sum_{r=0 to d-1} (N/d) = Sum_{d=1 to N} N = N^2.
    
    # Let's use a generator expression inside max().
    ans = max(
        max(
            (
                max(
                    [len(list(g)) for k, g in groupby(sorted([h[i] for i in range(r, n, d)]))]
                )
                for r in range(d)
            ),
            1
        )
        for d in range(1, n)
    )
    print(ans)

if __name__ == "__main__":
    main()