import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_time tracks the total height added to all plants since the start.
    # When a plant is planted at time 't', its "base height" is -t.
    # Its height at any time T is: (T - t).
    # We want to harvest plants where (T - t) >= H, which means t <= T - H.
    
    current_time = 0
    # Use a min-priority queue to store the 'planting times' of the plants.
    # The plant planted earliest has the smallest 't' and thus the largest height.
    # Wait, to harvest plants with height >= H, we need those with the smallest t.
    # Actually, if height is (current_time - planting_time), then
    # current_time - planting_time >= H  =>  planting_time <= current_time - H.
    # We store the planting times in a min-heap to easily extract the smallest ones.
    
    pq = []
    
    results = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant a plant with height 0.
            # Its height at time T is T - current_time.
            # We store the 'effective' birth time.
            heapq.heappush(pq, current_time)
            
        elif q_type == 2:
            T_val = int(input_data[ptr])
            ptr += 1
            current_time += T_val
            
        elif q_type == 3:
            H_val = int(input_data[ptr])
            ptr += 1
            
            # Harvest plants where height >= H
            # height = current_time - birth_time
            # current_time - birth_time >= H  =>  birth_time <= current_time - H
            threshold = current_time - H_val
            
            count = 0
            while pq and pq[0] <= threshold:
                heapq.heappop(pq)
                count += 1
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == '__main__':
    solve()