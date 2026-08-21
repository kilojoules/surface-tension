import sys

def solve():
    # Read N from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We want to find the maximum number of buildings that:
    # 1. Have the same height.
    # 2. Are at equal intervals.
    
    # Group indices by height to reduce the search space
    height_groups = {}
    for i in range(n):
        height = h[i]
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(i)
    
    max_buildings = 0
    
    # Iterate through each unique height present in the buildings
    for height in height_groups:
        indices = height_groups[height]
        num_indices = len(indices)
        
        # If the total number of buildings of this height is less than 
        # the current max, we can't possibly improve the answer here.
        if num_indices <= max_buildings:
            continue
            
        # Try every pair of buildings of the same height as the starting 
        # point and the second point to determine the interval 'd'.
        for i in range(num_indices):
            start_idx = indices[i]
            for j in range(i + 1, num_indices):
                second_idx = indices[j]
                d = second_idx - start_idx
                
                # Count how many buildings of the same height follow this interval
                count = 2
                current_idx = second_idx + d
                
                # We use a while loop to check for subsequent buildings at interval d.
                # Since we only care about buildings of the same height, 
                # we check if the building at current_idx exists and has the same height.
                while current_idx < n:
                    if h[current_idx] == height:
                        count += 1
                        current_idx += d
                    else:
                        break
                
                if count > max_buildings:
                    max_buildings = count
                    
    # If no pairs were found (all heights unique or N=1), the answer is 1.
    if max_buildings == 0 and n > 0:
        max_buildings = 1
        
    print(max_buildings)

if __name__ == "__main__":
    solve()