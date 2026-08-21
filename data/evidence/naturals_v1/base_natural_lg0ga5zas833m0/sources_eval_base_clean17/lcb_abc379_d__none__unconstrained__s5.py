import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_height_offset tracks the total growth time passed since the start.
    # When a plant is added at time 'now', its relative birth height is -current_height_offset.
    # Its actual height at any time is: current_height_offset + birth_height.
    current_height_offset = 0
    
    # We use a min-priority queue to store the birth heights of the plants.
    # This allows us to efficiently find and remove plants that have reached height H.
    # Condition: current_height_offset + birth_height >= H  =>  birth_height >= H - current_height_offset.
    # Wait, the min-heap should store birth heights so we can pop the smallest ones? 
    # No, the condition is "height at least H". 
    # Let's re-evaluate: 
    # Plant height = (Total growth since start) - (Total growth at time of planting).
    # Let S be the cumulative sum of T from type 2 queries.
    # Plant i planted at time t_i has height: S_now - S_{t_i}.
    # Harvest if: S_now - S_{t_i} >= H  => S_{t_i} <= S_now - H.
    # We want to count and remove all plants where S_{t_i} is small.
    # A min-heap of S_{t_i} values is perfect here.
    
    birth_times_heap = []
    
    results = []
    
    for _ in range(Q):
        query_type = int(input_data[ptr])
        ptr += 1
        
        if query_type == 1:
            # Plant height 0 means its relative birth time is the current offset
            heapq.heappush(birth_times_heap, current_height_offset)
            
        elif query_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_height_offset += T
            
        elif query_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            # Harvest plants where: current_height_offset - birth_time >= H
            # which is: birth_time <= current_height_offset - H
            threshold = current_height_offset - H
            count = 0
            while birth_times_heap and birth_times_heap[0] <= threshold:
                heapq.heappop(birth_times_heap)
                count += 1
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()