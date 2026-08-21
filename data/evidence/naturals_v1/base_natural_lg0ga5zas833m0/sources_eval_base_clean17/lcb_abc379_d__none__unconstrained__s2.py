import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # We need to track the current "global height" added to all plants.
    # When a plant is added at time t, its initial height is 0.
    # To treat it as if it grows with the global height, we store it as 
    # (height - current_global_height).
    # Since initial height is 0, we store it as -current_global_height.
    
    current_global_height = 0
    # Min-heap to store the "normalized" heights of plants.
    # Normalized height = actual height - current_global_height.
    # A plant is harvested if actual height >= H, 
    # which means (normalized height + current_global_height) >= H,
    # or normalized height >= H - current_global_height.
    plants = []
    
    results = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant height 0. Normalized height = 0 - current_global_height
            heapq.heappush(plants, -current_global_height)
        elif q_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_global_height += T
        elif q_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            # Threshold for normalized height
            threshold = H - current_global_height
            
            count = 0
            # Harvest all plants whose normalized height is >= threshold.
            # Wait, the heap gives us the SMALLEST normalized height.
            # We need to harvest plants with height >= H.
            # This means we want to remove plants from the heap that are "large".
            # However, the condition is "height >= H".
            # The plants that are EASIEST to harvest are those with the SMALLEST 
            # normalized height? No, that's wrong.
            # The plants that are EASIEST to harvest are those that were planted 
            # earliest (they have the smallest normalized height values, 
            # as they were subtracted by a smaller global height).
            
            # Let's re-evaluate:
            # Plant 1: added at global=0. Norm = 0.
            # Plant 2: added at global=10. Norm = -10.
            # Current global = 15.
            # Plant 1 actual height = 0 + 15 = 15.
            # Plant 2 actual height = 0 + (15 - 10) = 5.
            # If H = 10, Plant 1 is harvested.
            # Norm heights: Plant 1 is 0, Plant 2 is -10.
            # Threshold = 10 - 15 = -5.
            # Plant 1: 0 >= -5 (True), Plant 2: -10 >= -5 (False).
            
            # So we harvest plants with normalized height >= threshold.
            # But a min-heap lets us remove the SMALLEST values.
            # To remove the LARGEST values, a min-heap is not helpful.
            # WAIT: The plants that are most likely to be harvested are the ones 
            # planted EARLIEST. Those have the HIGHEST normalized heights.
            # Let's check: Plant 1 (Norm 0), Plant 2 (Norm -10). 
            # 0 is larger than -10.
            # So we need a MAX-heap to efficiently find plants with normalized height >= threshold.
            pass

# Correction: Use a max-heap to store normalized heights.
# In Python, heapq is a min-heap, so we store negative values to simulate a max-heap.

def solve_fixed():
    import sys
    import heapq
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    current_global_height = 0
    # Max-heap of normalized heights. Store as -val to use heapq.
    # Normalized height = actual_height - current_global_height
    # Initially actual_height = 0, so norm = -current_global_height.
    plants = []
    
    output = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # normalized height = -current_global_height
            # push -(-current_global_height) = current_global_height
            heapq.heappush(plants, current_global_height)
        elif q_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_global_height += T
        elif q_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            threshold = H - current_global_height
            # We harvest plants where normalized_height >= threshold.
            # Our heap stores -normalized_height.
            # So we harvest where -normalized_height <= -threshold.
            # The smallest value in the min-heap is the largest normalized height.
            
            count = 0
            while plants and (-plants[0]) >= threshold:
                heapq.heappop(plants)
                count += 1
            output.append(str(count))
            
    sys.stdout.write("\n".join(output) + "\n")

if __name__ == "__main__":
    solve_fixed()