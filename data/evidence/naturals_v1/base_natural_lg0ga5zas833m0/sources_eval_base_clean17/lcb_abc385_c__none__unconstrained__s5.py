import sys

def solve():
    # Read N from standard input
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
    
    # For each height group, check all possible intervals
    for height in height_groups:
        indices = height_groups[height]
        num_indices = len(indices)
        
        # If the total number of buildings of this height is less than 
        # or equal to the current max, we can't improve the result.
        if num_indices <= max_buildings:
            continue
            
        # We check every pair of indices (i, j) as the start and second element
        # to define a potential equal interval 'd'.
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                # Interval d between the i-th and j-th building of this height
                d = indices[j] - start_idx
                
                # Count how many buildings of the same height exist at interval d
                count = 0
                current_idx = start_idx
                # Since we know indices[j] is at start_idx + d, 
                # we can start counting from start_idx and jump by d.
                # To optimize, we can use a set for O(1) lookup.
                
                # We create a set for the current height's indices for fast lookup
                # (Moving this outside the j loop for efficiency)
                pass
        
        # Optimized inner loop:
        idx_set = set(indices)
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                d = indices[j] - start_idx
                
                # We only check this interval if it's possible to beat max_buildings
                # A rough upper bound is (n - 1 - start_idx) // d + 1
                if (n - 1 - start_idx) // d + 1 <= max_buildings:
                    continue
                
                count = 0
                curr = start_idx
                while curr < n:
                    if curr in idx_set:
                        count += 1
                    else:
                        # Condition: "The chosen buildings all have the same height"
                        # If we encounter a building of a different height at the 
                        # calculated interval, this specific sequence is broken.
                        # However, the problem asks for "some" buildings.
                        # Actually, the condition "arranged at equal intervals" 
                        # means if we pick indices i, i+d, i+2d... they must all be height H.
                        # If one is not, we cannot include it, and since they must be 
                        # equal intervals, we cannot skip it.
                        break
                    curr += d
                
                if count > max_buildings:
                    max_buildings = count
                    
    # Handle case where N=1 or no pairs were found
    if max_buildings == 0 and n > 0:
        max_buildings = 1
        
    print(max_buildings)

if __name__ == "__main__":
    solve()