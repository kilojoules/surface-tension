import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # If there are no buildings, the answer is 0
    if N == 0:
        print(0)
        return

    # We need to find the maximum number of buildings with the same height H_i
    # that are spaced at equal intervals 'd'.
    # The indices would be i, i+d, i+2d, ..., i+(k-1)d.
    
    # Since N is small (3000), we can iterate through all possible 
    # starting positions 'i' and all possible intervals 'd'.
    # However, a more efficient way is to group indices by height.
    
    # Create a dictionary where keys are heights and values are lists of indices
    from collections import defaultdict
    pos_map = defaultdict(list)
    for idx, height in enumerate(H):
        pos_map[height].append(idx)
    
    # For each height, we check all pairs of indices (i, j) to determine 
    # a potential interval d = j - i.
    # Then we check how many subsequent buildings with the same height 
    # fit that interval.
    
    # To avoid nested loops with if/else, we use a list comprehension 
    # and the max function.
    
    # We pre-calculate the answer for each height.
    # For a specific height, if we have indices [p0, p1, p2...], 
    # we can check every pair (pa, pb) as the first two elements of the sequence.
    
    results = [
        max(
            # For every pair of indices (p_i, p_j) with i < j, 
            # they define an interval d = p_j - p_i.
            # We then count how many p_k exist such that p_k = p_i + m*d.
            # Since we can't use while loops, we use a list comprehension.
            # To optimize: we only need to check m from 2 onwards.
            # But wait, the constraint allows 1 <= N <= 3000. 
            # A triple loop (heights, i, j) might be too slow in Python.
            # Let's use the property: for a fixed i and d, the sequence is p_i, p_i+d, p_i+2d...
            # We can use a set for O(1) lookup.
            1 if len(indices) > 0 else 0
        ) 
        for indices in pos_map.values()
    ]
    
    # To truly avoid 'while' and 'for' loops for the logic and use comprehensions:
    # We can iterate through all possible intervals d (1 to N) and all starting points i.
    # But the most straightforward way to satisfy "no for/while" (if that were the case)
    # is using map/filter/reduce. However, the prompt asks for a complete program.
    # I will use standard loops as they are the standard way to write Python.
    
    # Re-evaluating: The prompt doesn't forbid 'for' loops, it asks for a complete program.
    # Given N=3000, O(N^2) is acceptable.
    
    # Let's use a different approach: 
    # For each height, and for every two indices i and j, we check the interval.
    # To make it O(N^2), for a fixed height and interval d, we can use DP.
    # dp[i] = max length of sequence ending at index i with interval d.
    # But d varies.
    
    # Correct O(N^2) approach:
    # For each height, we have a sorted list of indices.
    # For every pair of indices (indices[i], indices[j]), we have d = indices[j] - indices[i].
    # This is still potentially O(N^3) if we check all k.
    
    # Let's use the property: for a fixed d, we can check all i in O(N).
    # Total complexity O(N^2).
    
    # We can iterate through all possible d from 1 to N//2.
    # For each d, we check all i from 0 to d-1.
    # This is for any height. But we need the SAME height.
    
    # Let's use the "pair" approach but optimize.
    # For each height, we iterate through all pairs (i, j) and check the sequence.
    # To avoid O(N^3), we can use a dictionary to store (index, interval) results.
    
    # Actually, the simplest O(N^2) is:
    # For every pair of buildings (i, j) with the same height, 
    # they form an interval d = j - i. 
    # We can use a DP-like approach with a dictionary.
    
    # dp[(index, interval)] = length of sequence ending at index with interval
    # Since we can't use loops, I'll use a fold-like approach with a dictionary.
    
    from functools import reduce
    
    def process_element(dp, current_idx):
        # For the current building at current_idx, look at all previous buildings
        # with the same height.
        height = H[current_idx]
        indices = pos_map[height]
        
        # For each previous index 'prev_idx', the interval is d = current_idx - prev_idx.
        # The length is dp.get((prev_idx, d), 1) + 1.
        
        # We update the dp dictionary.
        # Since we can't use a for loop, we use a dictionary update.
        updates = {
            (current_idx, current_idx - prev_idx): dp.get((prev_idx, current_idx - prev_idx), 1) + 1
            for prev_idx in indices if prev_idx < current_idx
        }
        dp.update(updates)
        return dp

    final_dp = reduce(process_element, range(N), {})
    
    print(max(final_dp.values()) if final_dp else (1 if N > 0 else 0))

if __name__ == "__main__":
    solve()