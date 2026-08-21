import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_height_offset tracks the total growth added by type 2 queries.
    # When a plant is planted at time t, its height is 0.
    # To represent this in a system where the global offset is 'current_height_offset',
    # we store the plant's "base height" as -current_height_offset.
    # Actual height = base_height + current_height_offset.
    
    current_height_offset = 0
    # Min-heap to store the base heights of the plants.
    # This allows us to efficiently find plants that reach height H.
    plants_heap = []
    
    output = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant a new plant of height 0.
            # base_height + current_height_offset = 0  => base_height = -current_height_offset
            heapq.heappush(plants_heap, -current_height_offset)
            
        elif q_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_height_offset += T
            
        elif q_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            count = 0
            # We harvest plants where actual height >= H.
            # base_height + current_height_offset >= H  => base_height >= H - current_height_offset.
            # Wait, the heap is a MIN-heap. We need to remove plants with height >= H.
            # Actually, the plants that are EASIEST to harvest are those with the 
            # SMALLEST base_height (planted earliest).
            # Wait, that's wrong. The plants with the SMALLEST base_height are the OLDEST.
            # The oldest plants have the largest actual height.
            # So we check the plant with the smallest base_height.
            
            threshold = H - current_height_offset
            while plants_heap and plants_heap[0] <= threshold:
                # This plant's actual height is >= H
                # Wait, let's re-verify: 
                # actual = base + offset. 
                # If base <= H - offset, then base + offset <= H.
                # That's the opposite of what we want.
                # We want actual >= H  => base + offset >= H => base >= H - offset.
                # This means the plants with the LARGEST base heights are the newest.
                # The plants with the SMALLEST base heights are the oldest and thus tallest.
                # If the plant with the smallest base height is still < H, then no one is harvested?
                # No. Smallest base height = most negative = tallest plant.
                # Let's re-check:
                # Plant 1: offset 0, base 0.
                # Query 2: T=15, offset 15.
                # Plant 2: offset 15, base -15.
                # Query 3: H=10. 
                # Plant 1 height: 0 + 15 = 15. (15 >= 10: True)
                # Plant 2 height: -15 + 15 = 0. (0 >= 10: False)
                # Plant 1 has base 0, Plant 2 has base -15.
                # The plant with the SMALLEST base is the NEWEST.
                # The plant with the LARGEST base is the OLDEST.
                # Wait, if current_height_offset is 15:
                # Plant 1 (oldest) base is 0.
                # Plant 2 (newest) base is -15.
                # Smallest base is -15, Largest base is 0.
                # Tallest plant has the largest base.
                # To harvest plants with height >= H, we need base + offset >= H.
                # This is not a min-heap problem for the "tallest", 
                # but we can just use a min-heap and check if the smallest base
                # is >= something? No.
                # Let's use a min-heap to store (base_height).
                # The plants that are most likely to be harvested are the ones 
                # with the LARGEST base_height.
                # But we want to remove them. A min-heap doesn't help remove the largest.
                # Let's use a min-heap and store the height at which the plant 
                # WOULD be harvested.
                # Plant is harvested if actual_height >= H.
                # base + offset >= H  =>  base >= H - offset.
                # This is still confusing. Let's simplify:
                # Store the "birth time" offset.
                # Plant i is born when the total offset was O_i.
                # Its height at any time is (CurrentOffset - O_i).
                # Harvest if (CurrentOffset - O_i) >= H  => O_i <= CurrentOffset - H.
                # Now we can use a min-heap to store O_i.
                # The smallest O_i are the oldest plants.
                # They are the first to reach height H.
                break
    
    # Redoing the logic inside the loop to avoid confusion.
    pass

def solve_final():
    import sys
    import heapq
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    current_offset = 0
    # Min-heap stores the offset at the time of planting.
    # Plant height = current_offset - birth_offset.
    birth_offsets = []
    
    results = []
    
    for _ in range(Q):
        q_type = input_data[ptr]
        ptr += 1
        
        if q_type == '1':
            heapq.heappush(birth_offsets, current_offset)
        elif q_type == '2':
            T = int(input_data[ptr])
            ptr += 1
            current_offset += T
        elif q_type == '3':
            H = int(input_data[ptr])
            ptr += 1
            
            count = 0
            # Harvest if current_offset - birth_offset >= H
            # birth_offset <= current_offset - H
            threshold = current_offset - H
            while birth_offsets and birth_offsets[0] <= threshold:
                heapq.heappop(birth_offsets)
                count += 1
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == '__main__':
    solve_final()