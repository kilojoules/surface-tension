import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Get all unique heights present in the buildings
    unique_heights = set(h)
    
    # We need to find the maximum number of buildings that:
    # 1. Have the same height 'ht'
    # 2. Are spaced by a constant interval 'd'
    # 3. Start at index 'i'
    
    # We use a nested list comprehension to evaluate all valid combinations:
    # - ht: each unique height available
    # - i: starting index (0 to n-1)
    # - d: interval (1 to n-1)
    
    # For a fixed height ht, start i, and interval d:
    # The indices are i, i+d, i+2d... 
    # We count how many of these indices j satisfy h[j] == ht.
    # However, the condition "arranged at equal intervals" implies we pick 
    # a sequence and ALL picked buildings must satisfy the height condition.
    # The problem asks for the maximum number of buildings we CAN choose.
    # This means for a fixed ht, i, and d, we check the sequence i, i+d, i+2d...
    # and count how many satisfy h[j] == ht. 
    # WAIT: The condition is "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices (i, i+d, i+2d...), they must all have height ht.
    # If one in the middle doesn't have height ht, we can't just skip it; 
    # the "equal interval" applies to the set of chosen buildings.
    # So for a fixed ht, i, and d, we count the longest contiguous chain 
    # starting at i with step d where all have height ht.
    
    # Actually, the simplest interpretation is: 
    # Pick a height 'ht', a start 'i', and a step 'd'.
    # The buildings are at indices i, i+d, i+2d... 
    # We count how many of these have height 'ht'.
    # But the condition "arranged at equal intervals" means the indices 
    # of the chosen buildings must be an arithmetic progression.
    # If we choose indices {p1, p2, ..., pk}, then p_{j+1} - p_j = d.
    # This means we are looking for the maximum k such that 
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d] == ht.
    
    # To implement this without loops:
    # For every pair of indices (i, j) where h[i] == h[j], they define a potential 
    # interval d = j - i. We then check how many subsequent buildings 
    # h[j+d], h[j+2d]... also have the same height.
    
    # Let's use a different approach:
    # For every possible height 'ht', and every possible interval 'd' (1 to N),
    # and every starting position 'i' (0 to d-1),
    # we can identify blocks of buildings of height 'ht'.
    
    # However, the most straightforward way to implement this is:
    # For every height 'ht' in unique_heights:
    #   For every start 'i' from 0 to N-1:
    #     For every interval 'd' from 1 to N-1:
    #       Count how many k >= 0 satisfy i + k*d < N and h[i + k*d] == ht
    #       BUT they must be consecutive in the arithmetic progression.
    #       Actually, the problem says "The chosen buildings are arranged at equal intervals."
    #       This means if you pick indices x_1 < x_2 < ... < x_k, 
    #       then x_2 - x_1 = x_3 - x_2 = ... = x_k - x_{k-1}.
    #       This is exactly an arithmetic progression.
    #       All these buildings must have the same height.
    
    # So for a fixed ht, i, and d, we want to find the maximum k such that
    # h[i] == ht, h[i+d] == ht, ..., h[i+(k-1)d] == ht.
    
    # Since N is small (3000), we can't do O(N^3). 
    # But we only need to check i and d such that h[i] == ht.
    # Let's refine:
    # For each height 'ht', find all indices where h[idx] == ht.
    # For every pair of such indices (i, j) with i < j, they define d = j - i.
    # Then we check how many more indices i + 2d, i + 3d... also have height 'ht'.
    
    # To avoid loops and O(N^3), we can use the following:
    # For a fixed height 'ht', let S be the set of indices where h[idx] == ht.
    # We want to find max k such that {i, i+d, ..., i+(k-1)d} is a subset of S.
    
    # We can use a dictionary/map to store the length of the progression 
    # ending at index j with difference d.
    # dp[j][d] = dp[j-d][d] + 1 if h[j] == h[j-d] else 1.
    
    # Since we can't use loops, we can use a reduction or a 
    # comprehension that builds the result.
    # But DP is hard without loops. Let's use the O(N^2) approach:
    # For every pair (i, j) with i < j and h[i] == h[j]:
    # the length is 2 + (count of k such that h[j + k*(j-i)] == h[i])
    
    # Given the constraints and the "no loop" rule, the most idiomatic 
    # way to handle this is using comprehensions.
    # To keep it O(N^2), we can iterate over all pairs (i, j) and 
    # for each pair, calculate the length of the progression.
    # Wait, if we check all pairs and then a while-loop for the rest, it's O(N^2).
    # But we can't use while-loops.
    
    # Let's use the property: for a fixed d, we can group indices by (i % d).
    # For a fixed d and a fixed remainder r, we have a sequence h[r], h[r+d], h[r+2d]...
    # We want the longest run of identical values in this sequence.
    
    # We can use `itertools.groupby` to find runs of identical heights.
    from itertools import groupby
    
    # We iterate over all possible intervals d from 1 to N-1.
    # For each d, we partition the array into d sequences.
    # For each sequence, we find the longest run of identical elements.
    
    # Use a generator expression to find the max length.
    # The structure:
    # max(
    #   max(
    #     len(list(group)) 
    #     for d in range(1, n)
    #     for r in range(d)
    #     for key, group in groupby(h[r::d])
    #   ),
    #   1
    # )
    
    # Note: h[r::d] is a slice, which is efficient.
    # groupby(h[r::d]) groups consecutive identical heights.
    
    # To handle the case where N=1, we ensure the range and max handle it.
    # The constraints say 1 <= N <= 3000.
    
    # We need to wrap this in a way that avoids 'for' loops.
    # The comprehension `for d in range(1, n) for r in range(d) ...` is allowed.
    
    # However, the nested max might be slow or hit recursion limits if not careful.
    # Let's use a single generator expression inside max().
    
    result = max([
        len(list(group))
        for d in range(1, n)
        for r in range(d)
        for key, group in groupby(h[r::d])
    ] + [1])
    
    print(result)

if __name__ == "__main__":
    solve()