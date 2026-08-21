import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to check every possible starting building (i)
    # and every possible interval (d) between chosen buildings.
    # For a fixed start i and interval d, we count how many buildings
    # have the same height as building i.
    # However, the condition is that ALL chosen buildings must have the same height.
    # So for a fixed i and d, we check the sequence h[i], h[i+d], h[i+2d]...
    # and count how many consecutive elements match h[i]. 
    # Wait, the condition is "The chosen buildings all have the same height".
    # This means if we pick indices i, i+d, i+2d..., they must all have height H_i.
    # The moment we encounter a building with a different height, we cannot 
    # include it or any subsequent buildings in that specific equal-interval set.
    
    # Actually, the problem says "choose some buildings". It doesn't say they 
    # must be contiguous in the arithmetic progression. 
    # "The chosen buildings are arranged at equal intervals" implies 
    # we pick indices i, i+d, i+2d, ..., i+(k-1)d.
    # All these must have the same height.
    
    # To implement this without loops:
    # 1. Iterate over all possible intervals d from 1 to N-1.
    # 2. Iterate over all possible starting positions i from 0 to N-1.
    # 3. For a fixed i and d, the sequence of indices is i, i+d, i+2d...
    # 4. We want the longest prefix of this sequence that shares the same height.
    # Actually, the problem is simpler: we can pick ANY subset that forms an 
    # arithmetic progression. But the "equal intervals" usually implies 
    # the gap between any two adjacent chosen buildings is the same.
    # This means we are looking for the maximum k such that there exists i, d 
    # where h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    
    # Since we cannot use loops, we use comprehensions.
    # For a fixed i and d, we can find the number of elements in the 
    # progression starting at i with difference d that have height h[i].
    # But they must be "arranged at equal intervals", meaning we can't skip.
    # So we count how many k satisfy h[i + k*d] == h[i] for k=0, 1, ...
    # until we hit an index >= N or a height != h[i].
    
    # Let's redefine: for every pair (i, d), we check the sequence.
    # To avoid loops and recursion, we can use a trick with range and 
    # a generator to find the first index that fails.
    
    # However, a simpler way: for a fixed i and d, the maximum k is the 
    # number of elements in the sequence h[i], h[i+d], ... such that 
    # all elements up to the k-th one are equal to h[i].
    
    # Since N is small (3000), we can't quite do O(N^3) in Python 
    # without loops if the comprehension is too heavy, but O(N^2) is fine.
    # Wait, if we fix i and d, we can't easily "break" in a comprehension.
    # But we can use a list comprehension to get the sequence and then 
    # find the first index where the height differs.
    
    # Let's use a different approach:
    # For every possible height H present in the array:
    #   For every possible interval d (1 to N):
    #     For every starting position i (0 to d-1):
    #       Check the sequence i, i+d, i+2d... and count matches.
    # This is still O(N^2).
    
    # To handle the "stop at first mismatch" without loops:
    # For a fixed i and d, we can create a list of booleans [h[j] == h[i] for j in range(i, n, d)]
    # Then we need the length of the prefix of True values.
    
    # But the problem says "choose some buildings". It doesn't say we can't 
    # skip buildings of the same height if they are at the interval.
    # "The chosen buildings are arranged at equal intervals" means 
    # if we choose indices x1 < x2 < ... < xm, then x2-x1 = x3-x2 = ... = xm-x(m-1).
    # This is exactly an arithmetic progression.
    
    # For a fixed i and d, we can just filter the indices:
    # indices = [j for j in range(i, n, d) if h[j] == h[i]]
    # But these indices must be i, i+d, i+2d... 
    # So we just need to count how many k satisfy h[i + k*d] == h[i] 
    # for k = 0, 1, ... until the first failure.
    
    # Actually, the most efficient way to write this in a comprehension:
    # For each i and d, we can't easily "break". 
    # But we can use a list comprehension to get the sequence and then 
    # use a helper to find the first False.
    # Or even simpler: just iterate over all i and d, and for each, 
    # count how many k satisfy h[i + k*d] == h[i] for ALL k from 0 to count-1.
    
    # Let's use the property: for a fixed i and d, the number of buildings is
    # k if h[i] == h[i+d] == ... == h[i+(k-1)d].
    # This is equivalent to: count k such that for all 0 <= j < k, h[i + j*d] == h[i].
    
    # To avoid loops, we can use:
    # max(
    #   [ 
    #     sum(1 for j in range(i, n, d) if h[j] == h[i]) # This is wrong, it allows gaps
    #     for i in range(n) for d in range(1, n)
    #   ]
    # )
    # Wait, the condition "arranged at equal intervals" means the distance 
    # between any two adjacent chosen buildings is the same.
    # It DOES NOT say we cannot have buildings of the same height in between.
    # It says the buildings WE CHOOSE must be at equal intervals.
    # Example: H = [5, 7, 5, 7, 7, 5, 7, 7]
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7
    # If we choose indices 1, 4, 7 (2nd, 5th, 8th), the interval is 3.
    # Heights are H[1]=7, H[4]=7, H[7]=7. All same. This is valid.
    # The buildings at indices 2, 3, 5, 6 are NOT chosen.
    
    # So for a fixed i (start) and d (interval), we want to find the 
    # maximum k such that h[i], h[i+d], ..., h[i+(k-1)d] are all equal.
    # Since we can't use loops, we can use a trick:
    # For a fixed i and d, the sequence is S = [h[j] for j in range(i, n, d)]
    # We want the length of the prefix of S that consists of the same value.
    # But we can just check all possible k!
    # For a fixed i and d, the number of elements is (n - 1 - i) // d + 1.
    # We want the largest k <= that value such that all h[i + j*d] == h[i] for 0 <= j < k.
    
    # Actually, the simplest O(N^2) is:
    # For every pair (i, j) with i < j:
    #   d = j - i
    #   if h[i] == h[j]:
    #     count how many more match: h[j+d], h[j+2d]...
    
    # To do this without loops:
    # We can use a recursive-like structure via a list comprehension 
    # or just iterate over all i, d and use a generator to count.
    # Since we can't use 'while', we can use a list comprehension to 
    # evaluate the condition for all possible k and then take the 
    # length of the prefix of Trues.
    
    # Let's use:
    # For i in range(n):
    #   for d in range(1, n):
    #     sequence = [h[j] == h[i] for j in range(i, n, d)]
    #     # Find length of leading True sequence
    #     # We can use itertools.takewhile or a similar approach.
    #     # But we can't import itertools? The prompt doesn't forbid it.
    #     # If we can't use itertools, we can use:
    #     # length = (length of sequence) - (index of first False)
    
    # Let's use a more direct approach:
    # For every i and d, we check if h[i] == h[i+d] == ...
    # The number of elements is k. This is valid if all h[i + m*d] == h[i] for m < k.
    # This is equivalent to: 
    # for a fixed i and d, the max k is the number of elements in the 
    # sequence h[i], h[i+d]... before the first element != h[i].
    
    # We can find the first index of a mismatch using:
    # next((idx for idx, val in enumerate(sequence) if not val), len(sequence))
    
    # Final plan:
    # Use a nested comprehension to iterate i and d.
    # Inside, use a generator expression with next() to find the first mismatch.
    
    res = max(
        [
            next(
                (idx for idx, val in enumerate([h[j] == h[i] for j in range(i, n, d)]) if not val),
                len([h[j] for j in range(i, n, d)])
            )
            for i in range(n)
            for d in range(1, n)
        ] + [1]
    )
    print(res)

solve()