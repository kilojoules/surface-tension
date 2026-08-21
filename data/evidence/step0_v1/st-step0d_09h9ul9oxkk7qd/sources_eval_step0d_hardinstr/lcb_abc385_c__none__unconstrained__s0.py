import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings with the same height 
    # placed at equal intervals.
    # An interval 'd' can range from 1 to n-1.
    # A starting position 'i' can range from 0 to n-1.
    
    # For a fixed start i and interval d, the buildings are at indices i, i+d, i+2d...
    # They satisfy the condition if h[i] == h[i+d] == h[i+2d]...
    
    # We can use a nested list comprehension to evaluate all pairs of (i, d).
    # For each pair, we count how many consecutive buildings starting from i 
    # with step d have the same height as h[i].
    
    # To avoid explicit loops, we use comprehensions.
    # We iterate over all possible starting indices i and all possible intervals d.
    # For each (i, d), we create a sequence of indices and count the prefix of matching heights.
    
    # However, a simpler way is: for every i and d, check the sequence 
    # h[i], h[i+d], h[i+2d]... and find the longest contiguous prefix of identical values.
    # But the problem says "the chosen buildings", implying we can pick a subset.
    # Actually, "arranged at equal intervals" means if we pick indices p1, p2, ..., pk,
    # then p_{j+1} - p_j = d for all j. This means they must be a contiguous 
    # arithmetic progression of indices.
    
    # Let's define a helper logic: for a fixed i and d, 
    # the number of buildings is the count of k such that i + k*d < n 
    # AND h[i + k*d] == h[i], but they must be consecutive in the progression.
    # Wait, the condition "arranged at equal intervals" means the distance between 
    # any two adjacent chosen buildings is the same. 
    # This means we are looking for the largest k such that there exists i, d 
    # where h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    
    # We can pre-calculate the lengths for all i, d.
    # Since we can't use loops, we use a comprehension.
    # For a fixed i and d, the number of buildings is:
    # we can use a trick with itertools.takewhile or just a comprehension that 
    # checks the condition.
    
    # Since we can't use while loops, we can use a list comprehension to 
    # get all indices [i, i+d, i+2d, ...] and then find the length of the 
    # prefix that matches h[i].
    
    # But wait, the problem doesn't say they must be a contiguous prefix.
    # "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices {p1, p2, ..., pk}, then p2-p1 = p3-p2 = ... = pk-p_{k-1} = d.
    # This is exactly an arithmetic progression.
    # For the condition to hold, h[p1] = h[p2] = ... = h[pk].
    
    # To find the max k for a fixed i and d:
    # We check indices i, i+d, i+2d... as long as they are < n and h[idx] == h[i].
    # Since we can't use while, we can use a list comprehension to get all 
    # indices in the progression and then find the first index that fails.
    
    # Actually, the simplest way:
    # For every i and d, we check how many k satisfy h[i + k*d] == h[i].
    # But they must be consecutive. So if h[i + k*d] != h[i], we stop.
    
    # Let's use a different approach:
    # For every pair (i, d), we want to find the largest k such that 
    # h[i] == h[i+d] == ... == h[i+(k-1)d].
    # We can use a recursive-like structure via a list comprehension or 
    # just iterate through all possible k and check.
    
    # Given N=3000, O(N^2) is acceptable. 
    # For every pair (i, d), we can't easily count the prefix without a loop.
    # But we can iterate over all i and d, and for each, 
    # calculate the length of the matching sequence.
    
    # Let's use the property: 
    # For a fixed d, we can group indices by (i % d).
    # For each group, we look for the longest run of identical heights.
    
    # To avoid loops, we use map, filter, and comprehensions.
    
    # For a fixed d:
    # We can create strings or tuples of heights for each offset r in 0...d-1.
    # Then find the longest run of identical elements in those sequences.
    
    # However, the simplest O(N^2) is:
    # For every i and d, check how many match.
    # But we need to stop at the first mismatch.
    
    # Let's use this: for every i and d, the number of buildings is 
    # the number of k such that for all 0 <= j < k, h[i + j*d] == h[i].
    # This is still hard without loops.
    
    # Let's reconsider: 
    # For every i and d, we can check all k from 1 up to N.
    # But that's O(N^3).
    
    # Correct O(N^2) approach:
    # For every d from 1 to N:
    #   For every i from 0 to d-1:
    #     Sequence S = [h[i], h[i+d], h[i+2d], ...]
    #     Find the longest run of identical values in S.
    
    # To find the longest run of identical values in a list S without loops:
    # We can use groupby from itertools.
    
    from itertools import groupby
    
    # We use a nested comprehension to iterate d and i.
    # For each d and i, we get the sequence S.
    # Then we use groupby to find the lengths of runs of identical elements.
    # Then we take the max of those lengths.
    
    # The total result is the max over all d and i.
    
    # Note: d can range from 1 to N. i from 0 to d-1.
    # The sequence S is h[i::d].
    
    # We need to handle the case where N=1 separately or ensure the logic covers it.
    # If N=1, the answer is 1.
    
    # The expression:
    # max(
    #   max(
    #     [len(list(g)) for k, g in groupby(h[i::d])]
    #     for i in range(d)
    #   )
    #   for d in range(1, n)
    # )
    # But we need to handle the case where the inner max might be empty or 
    # the outer max might be empty.
    # Also, we must include the case where we just pick 1 building.
    
    # Since we can't use loops, we can use a generator expression inside max().
    # We add 1 as a default value to max().
    
    # To avoid the "no loop" restriction while keeping it clean:
    # We can use a single nested comprehension.
    
    res = max([1] + [
        len(list(g))
        for d in range(1, n)
        for i in range(d)
        for k, g in groupby(h[i::d])
    ])
    
    print(res)

if __name__ == "__main__":
    solve()