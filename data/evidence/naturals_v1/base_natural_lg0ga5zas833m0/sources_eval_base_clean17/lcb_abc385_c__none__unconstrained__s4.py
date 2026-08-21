import sys

def solve():
    # Read N from the first line
    line1 = sys.stdin.readline()
    if not line1:
        return
    n = int(line1.strip())
    
    # Read the heights from the subsequent line(s)
    h = []
    while len(h) < n:
        line = sys.stdin.readline()
        if not line:
            break
        h.extend(map(int, line.split()))

    # Group indices of buildings by their height
    # height_map[height] = [index1, index2, ...]
    height_map = {}
    for i in range(n):
        height = h[i]
        if height not in height_map:
            height_map[height] = []
        height_map[height].append(i)

    max_buildings = 0

    # For each unique height, we check all possible pairs of indices
    # to determine the interval 'd' and calculate how many buildings 
    # of that height exist at that interval.
    for height in height_map:
        indices = height_map[height]
        num_indices = len(indices)
        
        # If the total buildings of this height is less than or equal to 
        # the current max, we can't improve the answer with this height.
        if num_indices <= max_buildings:
            continue
        
        # Every single building is a valid set of size 1.
        if num_indices >= 1:
            max_buildings = max(max_buildings, 1)
            
        # Check every pair of buildings of the same height to define a potential interval.
        # Let the first building be at index 'i' and the second at index 'j'.
        # The interval is d = j - i.
        for i in range(num_indices):
            idx_i = indices[i]
            for j in range(i + 1, num_indices):
                idx_j = indices[j]
                d = idx_j - idx_i
                
                # Count how many buildings of the same height exist at this interval.
                # We start from the first building (idx_i) and jump by d.
                count = 0
                current_idx = idx_i
                # Using a while loop to check indices in the original height array.
                # We only care if the height matches.
                while current_idx < n:
                    if h[current_idx] == height:
                        count += 1
                    else:
                        # The condition is: "The chosen buildings all have the same height."
                        # If we encounter a building at the interval that doesn't match
                        # the height, this specific sequence is interrupted.
                        # However, the problem asks for "chosen buildings", meaning 
                        # we can just skip the ones that don't match? 
                        # RE-READ: "The chosen buildings are arranged at equal intervals."
                        # This means if we pick indices x, x+d, x+2d..., ALL must have height H.
                        # So if h[current_idx] != height, this sequence is invalid.
                        count = 0 
                        break
                    current_idx += d
                
                if count > max_buildings:
                    max_buildings = count

    # Re-evaluating the interval logic: 
    # The problem says "chosen buildings are arranged at equal intervals".
    # This implies if we pick indices i, i+d, i+2d... i+(k-1)d, 
    # then h[i] == h[i+d] == ... == h[i+(k-1)d].
    # My nested loop approach above:
    # For each pair (i, j), d = j - i. We check i, i+d, i+2d...
    # This is O(N^3) in worst case, but N=3000 is too large for O(N^3).
    # Let's optimize.
    
    # Optimized approach:
    # For each height, iterate through its indices.
    # For each index i, and each possible interval d (1 to N), 
    # check how many match. But d can be anything.
    # Actually, for a fixed height and a fixed starting index i, 
    # and a fixed interval d, the number of buildings is limited.
    
    # Let's refine the logic:
    # For each height:
    #   indices = list of indices with that height
    #   for i in range(len(indices)):
    #     for j in range(i + 1, len(indices)):
    #       d = indices[j] - indices[i]
    #       # check indices[i], indices[i]+d, indices[i]+2d...
    #       # This is still potentially slow.
    
    # Wait, the number of buildings for a fixed d is at most N/d.
    # Total complexity: Sum_{heights} (num_indices^2 * (N/d))
    # This is still potentially slow. Let's implement it cleanly.

def solve_final():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    height_map = {}
    for i in range(n):
        val = h[i]
        if val not in height_map:
            height_map[val] = []
        height_map[val].append(i)

    ans = 0
    for val in height_map:
        indices = height_map[val]
        if len(indices) <= ans:
            continue
        
        # If we have at least one building of this height
        ans = max(ans, 1)
        
        # Try all pairs to define the interval
        for i in range(len(indices)):
            start_idx = indices[i]
            for j in range(i + 1, len(indices)):
                d = indices[j] - start_idx
                
                # Optimization: if the maximum possible buildings with this d 
                # is less than current ans, skip.
                # Max possible = (n - 1 - start_idx) // d + 1
                if (n - 1 - start_idx) // d + 1 <= ans:
                    continue
                
                count = 0
                curr = start_idx
                while curr < n:
                    if h[curr] == val:
                        count += 1
                        curr += d
                    else:
                        break
                ans = max(ans, count)
    
    print(ans)

if __name__ == '__main__':
    solve_final()