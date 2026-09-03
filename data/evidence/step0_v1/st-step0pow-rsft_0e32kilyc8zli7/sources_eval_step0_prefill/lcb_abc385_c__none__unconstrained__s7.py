import sys

def solve():
    # Read N from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices of buildings by their height
    # height_groups[height] = [index1, index2, ...]
    height_groups = {}
    for i in range(n):
        height = h[i]
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(i)
    
    max_buildings = 0
    
    # For each height, check all possible starting positions and intervals
    for height in height_groups:
        indices = height_groups[height]
        num_indices = len(indices)
        
        # If the total number of buildings of this height is less than 
        # the current max, we can't possibly beat the record.
        if num_indices <= max_buildings:
            continue
            
        # We need to check if we can pick buildings at equal intervals.
        # An interval is defined by the distance between two buildings of the same height.
        # Let the indices of buildings of the same height be sorted (they already are).
        # We try every pair of buildings as the first two in the sequence.
        for i in range(num_indices):
            for j in range(i + 1, num_indices):
                # First building at indices[i], second at indices[j]
                # Interval d = indices[j] - indices[i]
                d = indices[j] - indices[i]
                count = 2
                
                # Check subsequent buildings at the same interval
                next_idx = indices[j] + d
                # We use a set for O(1) lookup of indices for the current height
                # But since we only care about the current height, 
                # we can just check if next_idx is within bounds and has the correct height.
                # However, a set of indices for the current height is faster.
                
                # To optimize, we pre-calculate the set for the current height
                # (This is done outside the i, j loops)
                pass
        
    # Re-implementing the loop with the set optimization
    max_buildings = 0
    for height in height_groups:
        indices = height_groups[height]
        idx_set = set(indices)
        num_indices = len(indices)
        
        if num_indices <= max_buildings:
            continue
            
        # Every single building is a valid sequence of length 1
        if max_buildings < 1:
            max_buildings = 1
            
        for i in range(num_indices):
            for j in range(i + 1, num_indices):
                d = indices[j] - indices[i]
                # Current sequence: indices[i], indices[j]
                # We check how many more we can get
                count = 2
                curr = indices[j] + d
                while curr < n and curr in idx_set:
                    count += 1
                    curr += d
                
                if count > max_buildings:
                    max_buildings = count
                    
    # Handle the case where N=1 or no pairs were found
    if max_buildings == 0 and n > 0:
        max_buildings = 1
        
    print(max_buildings)

if __name__ == "__main__":
    solve()