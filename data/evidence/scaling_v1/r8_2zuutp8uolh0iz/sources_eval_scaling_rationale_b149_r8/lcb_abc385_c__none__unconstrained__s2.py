import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to check every possible starting building 'i'
    # and every possible interval 'd' between buildings.
    # For a fixed start 'i' and interval 'd', we count how many
    # buildings at indices i, i+d, i+2d... have the same height as h[i].
    
    # The range of i is 0 to n-1
    # The range of d is 1 to n-1
    # For a given i and d, the sequence of indices is range(i, n, d)
    
    # We use a generator expression inside max() to find the maximum count.
    # We handle the case where N=1 separately or ensure the range logic covers it.
    
    # result = max(
    #     count of buildings with height h[i] in the sequence starting at i with step d
    #     for i in range(n)
    #     for d in range(1, n)
    # )
    
    # To count the valid buildings in the sequence, we can use a list comprehension
    # and len(), or sum(1 for ...). 
    # However, the condition is that ALL chosen buildings must have the same height.
    # The problem asks for the maximum number of buildings we CAN choose.
    # This means for a fixed i and d, we can only pick buildings that match h[i].
    # Wait, the condition is "The chosen buildings all have the same height" AND 
    # "arranged at equal intervals". 
    # This means if we pick indices i, i+d, i+2d, we must check if h[i] == h[i+d] == h[i+2d].
    # If one in the middle doesn't match, we can't just skip it and keep the interval.
    # The "equal intervals" refers to the indices of the chosen buildings.
    # So if we choose indices (p1, p2, ..., pk), then p_{j+1} - p_j = d for all j.
    # This implies we are looking for the longest arithmetic progression of indices
    # such that all corresponding heights are identical.
    
    # For a fixed start i and interval d, we can take buildings at i, i+d, i+2d...
    # as long as they all have height h[i]. The moment we hit a building with a 
    # different height, we cannot include it, but we also cannot include any 
    # subsequent buildings because the "equal interval" must be maintained 
    # across the chosen set. 
    # Actually, the problem says "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices {i, i+d, i+2d, ..., i+(k-1)d}, 
    # then h[i] == h[i+d] == ... == h[i+(k-1)d].
    
    # Let's refine: for every pair (i, d), we want to find the largest k such that
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    # Since we can't use while loops, we can use a list comprehension to get all 
    # indices in the sequence and then find the first index that fails.
    # Or more simply: for a fixed i and d, we check all k such that i+(k-1)d < n.
    # The number of buildings is the number of elements in the sequence 
    # [h[i], h[i+d], h[i+2d], ...] that are equal to h[i] BEFORE the first mismatch.
    
    # Actually, the simplest way:
    # For every i and d, we look at the sequence H_i, H_{i+d}, H_{i+2d}...
    # We want to find the longest prefix of this sequence where all elements are equal.
    
    # But wait, the constraint to avoid loops makes "finding the first mismatch" 
    # tricky without recursion or while. 
    # Let's re-read: "The chosen buildings are arranged at equal intervals."
    # This means we pick a set of indices {p1, p2, ..., pk} such that p_{j+1} - p_j = d.
    # This is exactly what I described.
    
    # Since N is small (3000), we can't do O(N^3). O(N^2) is fine.
    # For a fixed i and d, we can't easily find the "prefix" length without a loop.
    # HOWEVER, we can just iterate over all possible k and check if the condition holds.
    # But that's O(N^3). 
    # Let's reconsider: for a fixed i and d, we can use a list comprehension to 
    # get the sequence and then use a trick to find the first mismatch.
    # Or, we can just iterate over all i and d, and for each, 
    # count how many elements in the sequence h[i::d] are equal to h[i].
    # WAIT: The problem does NOT say we must take a contiguous prefix.
    # It says "The chosen buildings are arranged at equal intervals."
    # If we choose indices 2, 5, 8, the interval is 3. We don't care about 3, 4, 6, 7.
    # We only care that the ones we PICKED are at equal intervals.
    # So for a fixed i and d, we can pick ALL indices j = i + m*d (where j < n)
    # such that h[j] == h[i].
    # NO, that's wrong. If we pick indices 2, 8, 14, the interval is 6.
    # If we pick 2, 5, 8, 11, the interval is 3.
    # The condition "arranged at equal intervals" means the gap between 
    # consecutive chosen buildings is constant.
    # So we pick indices: i, i+d, i+2d, ..., i+(k-1)d.
    # ALL of these must have the same height.
    # This means we are looking for the maximum k such that there exists i, d 
    # where h[i] = h[i+d] = ... = h[i+(k-1)d].
    
    # To solve this without loops/recursion:
    # For every i and d, we can check the sequence h[i::d].
    # We want the longest prefix of identical values.
    # Since we can't use while, we can use a list comprehension to find all 
    # indices where the value differs from h[i], and use that to find the prefix length.
    
    # Actually, the most straightforward O(N^2) approach:
    # For every pair of indices (i, j) with i < j, they define a potential 
    # height H = h[i] and an interval d = j - i.
    # We can then check how many further buildings h[j+d], h[j+2d]... also have height H.
    
    # Let's use the property: for a fixed i and d, the number of buildings is:
    # k = 1 + (number of consecutive elements in h[i+d::d] that equal h[i])
    
    # To implement "count consecutive" without loops:
    # For a fixed i and d, we can create a list of booleans [h[j] == h[i] for j in range(i, n, d)]
    # Then we need the length of the leading True sequence.
    
    # Given N=3000, O(N^2) is 9 million. Python might be slow.
    # Let's optimize: iterate over all possible heights H (up to 3000).
    # For each height, find all indices where h[i] == H.
    # Then for every pair of those indices, check the interval.
    
    # Actually, the simplest O(N^2) is:
    # For every d from 1 to N:
    #   For every i from 0 to d-1:
    #     Process the sequence h[i], h[i+d], h[i+2d]...
    #     In this sequence, find the longest run of identical values.
    
    # To find the longest run of identical values in a list without loops:
    # We can use itertools.groupby.
    
    from itertools import groupby
    
    # We can use a nested comprehension to evaluate all possible d and i.
    # For each (d, i), we get the sequence h[i::d].
    # Then we use groupby to find lengths of runs of identical heights.
    # Then we take the max length.
    
    # To avoid explicit loops, we use map/max/comprehensions.
    
    # The total number of elements processed across all d, i is:
    # sum_{d=1}^{N-1} sum_{i=0}^{d-1} (N/d) = sum_{d=1}^{N-1} N = N(N-1)
    # This is O(N^2).
    
    # We need to handle N=1 case.
    if n == 0:
        print(0)
        return

    # Use a generator to keep memory low and max() to find the result.
    # We iterate d from 1 to n-1, and i from 0 to d-1.
    # For each, we look at the sequence h[i::d].
    # We group identical consecutive elements and find the max length of a group.
    
    # Since we can't use for loops, we use nested comprehensions.
    # The result is the maximum length of any group across all d and i.
    
    # Note: if N=1, the range(1, n) is empty, so we provide a default of 1.
    
    ans = max([1] + [
        max([len(list(g)) for k, g in groupby(h[i::d])])
        for d in range(1, n)
        for i in range(d)
    ])
    
    print(ans)

solve()