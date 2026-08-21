import sys

def solve():
    # Read N from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The constraints are N <= 3000, H_i <= 3000.
    # A naive O(N^3) approach might be too slow, but we can optimize.
    # For every pair of buildings (i, j) with the same height, 
    # they define a potential interval d = j - i.
    # We can check how many buildings of that same height exist at intervals of d.
    
    # To optimize, we can group indices by height.
    height_map = {}
    for i in range(N):
        h = H[i]
        if h not in height_map:
            height_map[h] = []
        height_map[h].append(i)
    
    max_count = 1
    
    # Iterate through each unique height
    for h in height_map:
        indices = height_map[h]
        num_indices = len(indices)
        
        if num_indices <= max_count:
            continue
            
        # For this height, check all possible intervals d
        # An interval d can range from 1 to N-1
        # However, we only care about d that are differences between indices of the same height.
        # To avoid O(N^3), we can iterate through all possible intervals d and check.
        # But wait, N=3000, N^2 is 9 million. O(N^2) is acceptable.
        
        # For a fixed height, we can check every starting point i and every interval d.
        # But it's faster to iterate through the indices list.
        for i in range(num_indices):
            # Try every other index j as the second building to determine the interval d
            for j in range(i + 1, num_indices):
                d = indices[j] - indices[i]
                
                # We want to count how many indices k = indices[i] + m*d also have height h.
                # Since we only care about indices in the 'indices' list, 
                # we can use a set for O(1) lookup.
                
                # Optimization: if the remaining possible buildings cannot beat max_count, skip.
                # (N - indices[i] - 1) // d + 1 is the max possible buildings for this d.
                if (N - 1 - indices[i]) // d + 1 <= max_count:
                    continue
                
                count = 1
                current_pos = indices[i] + d
                # We check if current_pos is in the set of indices for height h.
                # To make this efficient, we use a boolean array or a set.
                # Using a set for the specific height:
                # (The set is created once per height)
                pass

    # Let's refine the logic for better performance.
    # We'll use a boolean array for height checks.
    # Since H_i is up to 3000, we can just use the H array directly.
    
    # Re-evaluating the approach:
    # For each height h that appears in the input:
    #   Find all positions where height is h.
    #   For every pair of positions (i, j), calculate d = j - i.
    #   Check how many positions i + k*d have height h.
    
    # Actually, a simpler O(N^2) is:
    # For every possible interval d from 1 to N-1:
    #   For every starting position i from 0 to d-1:
    #     Check sequences: i, i+d, i+2d...
    #     Within such a sequence, find the most frequent height.
    
    # Let's try the "most frequent height in sequence" approach:
    # For d in 1...N-1:
    #   For i in 0...d-1:
    #     counts = {}
    #     for k in range(i, N, d):
    #       h = H[k]
    #       counts[h] = counts.get(h, 0) + 1
    #       if counts[h] > max_count: max_count = counts[h]
    
    # Complexity: Sum_{d=1}^{N-1} Sum_{i=0}^{d-1} (N/d) = Sum_{d=1}^{N-1} N = N(N-1)
    # This is O(N^2), which is perfect for N=3000.

    max_count = 1
    for d in range(1, N):
        for i in range(d):
            # We only need to check if this sequence could possibly beat max_count
            # The length of the sequence is (N - 1 - i) // d + 1
            if (N - 1 - i) // d + 1 <= max_count:
                continue
                
            # Count heights in the arithmetic progression
            # Using a dictionary might be slow, but let's see.
            # Since H_i <= 3000, a fixed size array could be faster, 
            # but we'd need to clear it.
            local_counts = {}
            for k in range(i, N, d):
                h = H[k]
                local_counts[h] = local_counts.get(h, 0) + 1
                if local_counts[h] > max_count:
                    max_count = local_counts[h]
                    
    print(max_count)

if __name__ == "__main__":
    solve()