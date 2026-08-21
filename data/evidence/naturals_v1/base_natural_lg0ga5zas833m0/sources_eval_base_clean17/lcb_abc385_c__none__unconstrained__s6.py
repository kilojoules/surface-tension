import sys

def solve():
    # Read N from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The maximum number of buildings we can choose
    max_buildings = 0
    
    # We iterate through every possible starting building i
    # and every possible interval d.
    # However, a more efficient approach is to group buildings by height.
    
    # height_map stores indices of buildings for each unique height
    height_map = {}
    for idx, height in enumerate(h):
        if height not in height_map:
            height_map[height] = []
        height_map[height].append(idx)
        
    # For each unique height, we check all possible intervals
    for height in height_map:
        indices = height_map[height]
        num_indices = len(indices)
        
        # If the number of buildings of this height is less than or equal to 
        # the current max, we can't possibly beat the record with this height.
        if num_indices <= max_buildings:
            continue
            
        # Check every pair of indices as the first and second building of the sequence
        # to determine the interval d.
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                # Interval d = distance between building i and building j
                d = indices[j] - start_idx
                
                # Count how many buildings of the same height exist at this interval
                count = 0
                current_pos = start_idx
                while current_pos < n:
                    # Check if the building at current_pos has the target height
                    if h[current_pos] == height:
                        count += 1
                    else:
                        # The condition says the CHOSEN buildings must have the same height.
                        # It does NOT say buildings in between must have a different height.
                        # But the chosen buildings MUST be at equal intervals.
                        # If the building at current_pos is NOT the target height, 
                        # it cannot be part of the chosen set.
                        # Since we are checking if we can pick buildings at intervals of d,
                        # and the building at current_pos is NOT the target height,
                        # this specific sequence (start_idx, d) is invalid for this height.
                        # Wait, the condition is: "The chosen buildings all have the same height"
                        # AND "The chosen buildings are arranged at equal intervals".
                        # This means if we pick index i, i+d, i+2d... they must ALL be height H.
                        # If one is not, that sequence is broken.
                        break 
                    current_pos += d
                
                if count > max_buildings:
                    max_buildings = count
                    
    # If max_buildings is still 0 (shouldn't happen given N >= 1), 
    # the answer is 1 as per the problem statement.
    if max_buildings == 0 and n > 0:
        max_buildings = 1
        
    print(max_buildings)

if __name__ == "__main__":
    solve()