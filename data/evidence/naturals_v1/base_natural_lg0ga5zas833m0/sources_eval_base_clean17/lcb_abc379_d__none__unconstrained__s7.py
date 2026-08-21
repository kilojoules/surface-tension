import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_time tracks the cumulative growth (T) added since the start.
    # When a plant is added at time 't', its relative height is -t.
    # Its actual height at time 'T_now' is T_now + relative_height.
    current_time = 0
    # Min-heap to store the relative heights of the plants.
    # Plants with the smallest relative height are the tallest.
    # Wait, that's wrong. Relative height = (actual height - current_time).
    # For a new plant, height is 0, so relative height is 0 - current_time.
    # A plant is harvested if (current_time + relative_height) >= H,
    # which means relative_height >= H - current_time.
    # To efficiently find plants with relative_height >= threshold, 
    # we need a data structure that allows us to remove elements.
    # However, the condition is relative_height >= H - current_time.
    # This means we want to remove plants with the LARGEST relative heights.
    # A Max-Heap is suitable here.
    
    # Actually, let's re-evaluate:
    # Plant added at time t (where t is cumulative T): height is 0.
    # At time t_now, height is t_now - t.
    # Harvest if t_now - t >= H  =>  t <= t_now - H.
    # We want to count and remove all plants added at time t such that t <= (t_now - H).
    # This means we need to track the 'birth times' (cumulative T at planting)
    # and remove those that are small enough. A Min-Heap of birth times is perfect.
    
    birth_times = []
    results = []
    
    for _ in range(Q):
        q_type = input_data[ptr]
        ptr += 1
        
        if q_type == '1':
            # Plant a new flower pot. Its birth time is the current cumulative growth.
            heapq.heappush(birth_times, current_time)
        elif q_type == '2':
            t_val = int(input_data[ptr])
            ptr += 1
            current_time += t_val
        elif q_type == '3':
            h_val = int(input_data[ptr])
            ptr += 1
            
            # Condition: current_time - birth_time >= h_val
            # birth_time <= current_time - h_val
            threshold = current_time - h_val
            count = 0
            while birth_times and birth_times[0] <= threshold:
                heapq.heappop(birth_times)
                count += 1
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == '__main__':
    solve()