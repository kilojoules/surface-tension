import sys
import heapq

def solve():
    # Using fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_time represents the total height added to all plants
    # since the start of the simulation.
    current_time = 0
    
    # We store the "birth time" of the plants. 
    # If a plant is planted at current_time = X, its height at time T is (T - X).
    # We want to harvest plants where height >= H, which means (T - X) >= H.
    # This is equivalent to X <= T - H.
    # To efficiently find plants with X <= threshold, we use a Min-Heap of birth times.
    birth_times = []
    
    results = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant a new plant. Its current height is 0.
            # Its relative birth time is the current total time elapsed.
            heapq.heappush(birth_times, current_time)
            
        elif q_type == 2:
            # Wait for T days. All existing plants grow by T.
            t_val = int(input_data[ptr])
            ptr += 1
            current_time += t_val
            
        elif q_type == 3:
            # Harvest plants with height >= H.
            # height = current_time - birth_time.
            # height >= H  =>  birth_time <= current_time - H.
            h_val = int(input_data[ptr])
            ptr += 1
            
            threshold = current_time - h_val
            count = 0
            
            # Extract all plants from the heap that were born at or before the threshold.
            while birth_times and birth_times[0] <= threshold:
                heapq.heappop(birth_times)
                count += 1
            
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()