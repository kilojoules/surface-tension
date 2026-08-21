import sys

def solve():
    # Read N from the first line of input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        n = int(line1.strip())
        
        # Read the heights from the following lines
        h = []
        while len(h) < n:
            line = sys.stdin.readline()
            if not line:
                break
            h.extend(map(int, line.split()))
    except ValueError:
        return

    # max_buildings will store the maximum number of buildings found
    max_buildings = 0
    
    # To optimize, we can group indices of buildings by their height
    # height_map: { height: [list of indices] }
    height_map = {}
    for i in range(n):
        height = h[i]
        if height not in height_map:
            height_map[height] = []
        height_map[height].append(i)
        
    # Iterate through each unique height present in the buildings
    for height in height_map:
        indices = height_map[height]
        num_indices = len(indices)
        
        # If the total number of buildings of this height is less than or equal 
        # to the current max_buildings, we can't improve the result here.
        if num_indices <= max_buildings:
            continue
            
        # We need to find the maximum subset of indices that form an arithmetic progression.
        # Since N is up to 3000, an O(N^2) approach is acceptable.
        # For each pair of buildings of the same height, assume they are part of the sequence.
        for i in range(num_indices):
            # Start with a single building
            # Note: The problem says if we choose one, it satisfies the condition.
            # So we initialize with 1.
            
            # Optimization: if the remaining indices for this height cannot beat max_buildings, break.
            if (num_indices - i) <= max_buildings:
                break
                
            for j in range(i + 1, num_indices):
                # Let the first building be at indices[i] and the second at indices[j]
                # The interval is d = indices[j] - indices[i]
                d = indices[j] - indices[i]
                count = 2
                
                # Check for subsequent buildings at intervals of d
                next_idx = indices[j] + d
                # We only need to check if next_idx is within range and has the same height
                # Using a set for O(1) lookup of indices for the current height
                # However, since we are already iterating over indices of the same height,
                # we can use a pointer or a set.
                
                # Let's use a set for faster lookup of indices for the specific height
                # (created once per height)
                
        # Re-implementing the inner loop with a set for efficiency
        indices_set = set(indices)
        for i in range(num_indices):
            if (num_indices - i) <= max_buildings:
                break
            for j in range(i + 1, num_indices):
                d = indices[j] - indices[i]
                count = 2
                curr = indices[j] + d
                while curr < n and curr in indices_set:
                    count += 1
                    curr += d
                if count > max_buildings:
                    max_buildings = count
                    
    # Handle the case where N=1 or only 1 building is chosen
    if max_buildings == 0 and n > 0:
        max_buildings = 1
    elif n == 0:
        max_buildings = 0
    
    # Edge case: if max_buildings is still 0 but there are buildings, 
    # the answer is at least 1.
    if n > 0 and max_buildings < 1:
        max_buildings = 1

    print(max_buildings)

if __name__ == "__main__":
    solve()