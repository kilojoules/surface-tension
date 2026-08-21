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
    # An interval 'd' can range from 1 to N-1.
    # A starting position 'i' can range from 0 to N-1.
    
    # For a fixed start i and interval d, the buildings are at indices:
    # i, i+d, i+2d, ... as long as the index < N.
    # They satisfy the condition if h[i] == h[i+d] == h[i+2d] ...
    
    # We can use a helper function or a comprehension to count how many 
    # consecutive buildings starting from i with interval d have the same height.
    # However, the condition is that ALL chosen buildings must have the same height.
    # This means if we pick a set at interval d starting at i, we only count 
    # those that match h[i]. But the problem says "the chosen buildings" 
    # must have the same height and be at equal intervals. 
    # This implies we pick a starting point i and an interval d, and we 
    # check how many indices (i + k*d) have the height h[i].
    # Wait, the condition "arranged at equal intervals" means the indices 
    # must be i, i+d, i+2d... 
    # If we skip one that doesn't match the height, they are no longer at 
    # equal intervals relative to the whole set.
    # Therefore, we are looking for the longest sequence i, i+d, i+2d... 
    # such that h[i] = h[i+d] = h[i+2d] = ...
    
    # Let's redefine: for every pair (i, d), we count how many k >= 0 
    # satisfy i + k*d < N AND h[i + k*d] == h[i].
    # IMPORTANT: The condition "arranged at equal intervals" means the 
    # indices chosen must be an arithmetic progression. 
    # It does NOT say we can't have buildings of different heights 
    # BETWEEN the chosen ones, but the chosen ones themselves must 
    # be at equal intervals.
    # Example: Indices 2, 5, 8 (interval 3). 
    # We just need h[2] == h[5] == h[8].
    
    # To avoid loops, we use nested comprehensions.
    # 1. Iterate over all possible intervals d from 1 to N.
    # 2. Iterate over all possible starting positions i from 0 to N-1.
    # 3. For each (i, d), count how many k satisfy (i + k*d < N) and (h[i + k*d] == h[i]).
    # Note: The k's must be contiguous (0, 1, 2...) to maintain the "equal interval" 
    # across the chosen set. If h[i+d] != h[i], the sequence breaks.
    
    # Actually, the problem says "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices {p1, p2, ..., pm}, then p_{j+1} - p_j = d for all j.
    # So we are looking for the maximum m such that there exists i and d where
    # h[i] = h[i+d] = h[i+2d] = ... = h[i+(m-1)d].
    
    # To implement this without loops:
    # For a fixed i and d, the number of buildings is the length of the 
    # prefix of the sequence [h[i], h[i+d], h[i+2d], ...] that equals h[i].
    
    # Since we can't use while loops to find the prefix length, we can 
    # use a trick with itertools.takewhile or just check all possible lengths m.
    # But the simplest way is: for a fixed i and d, 
    # the count is the number of k such that for all 0 <= j <= k, h[i+j*d] == h[i].
    
    # Let's use a different approach:
    # For every i and d, we want to find the largest m such that 
    # h[i] == h[i+d] == ... == h[i+(m-1)d].
    # This is equivalent to:
    # m = 1 + (number of k > 0 such that h[i] == h[i+d] == ... == h[i+kd])
    
    # Since N is small (3000), we can't do O(N^3). O(N^2) is required.
    # We can iterate over all d and i, and for each, we need the length of the 
    # matching sequence.
    # To avoid loops and recursion, we can use a list comprehension to 
    # pre-calculate matches and then use a logic to find the streak.
    # But wait, the simplest O(N^2) is:
    # For each d in 1...N:
    #   For each i in 0...d-1:
    #     Process the sequence h[i], h[i+d], h[i+2d]...
    #     Find the longest run of identical values.
    
    # To do this without for/while:
    # We can use groupby from itertools to find runs of identical heights.
    from itertools import groupby
    
    # We generate all sequences for all d and i, then find the max group length.
    # The sequences are: [h[i::d] for d in range(1, n) for i in range(d)]
    # Then for each sequence, we find the max length of a group of identical elements.
    
    # Use a generator expression inside max() to keep memory low.
    # The nested structure:
    # max( 
    #   max( 
    #     len(list(group)) 
    #     for val, group in groupby(h[i::d])
    #   ) 
    #   for d in range(1, n) 
    #   for i in range(d)
    # )
    
    # Handle the case N=1 separately or ensure the range/max handles it.
    # If N=1, the range(1, 1) is empty. We should start with a default of 1.
    
    ans = max([1] + [
        len(list(group))
        for d in range(1, n)
        for i in range(d)
        for val, group in groupby(h[i::d])
    ])
    
    print(ans)

if __name__ == "__main__":
    solve()