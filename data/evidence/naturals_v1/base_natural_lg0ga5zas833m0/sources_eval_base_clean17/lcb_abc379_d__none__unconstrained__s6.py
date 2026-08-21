import sys
import heapq

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    ptr = 1
    
    # current_time tracks the cumulative height added by all type 2 queries.
    # When a plant is planted at time 't', its "birth height" relative to 
    # the current_time is -current_time.
    # A plant's actual height at any moment is (current_time + birth_height).
    # To harvest plants with height >= H, we need:
    # current_time + birth_height >= H  =>  birth_height >= H - current_time.
    # However, since we want to harvest the smallest birth_heights that satisfy this,
    # and birth_heights are added over time, it's easier to store 
    # the required birth_height to be harvested.
    # Let's use a min-priority queue to store the birth_heights.
    # A plant planted at current_time has birth_height = -current_time.
    # Height at time 'now' = now + birth_height.
    # Condition: now + birth_height >= H  =>  birth_height >= H - now.
    # This is not ideal for a min-heap because we want to remove elements
    # that are GREATER than or equal to a value.
    # Wait, the condition is: height >= H. 
    # If we store birth_heights, the plants that are "oldest" (smallest birth_height)
    # are the tallest.
    # Let's redefine: birth_height = the value of current_time when the plant was planted.
    # Height of plant = current_time - birth_height.
    # Condition: current_time - birth_height >= H  =>  birth_height <= current_time - H.
    # Now we can use a min-heap to store birth_heights. 
    # The smallest birth_height corresponds to the tallest plant.
    
    pq = []
    current_time = 0
    
    results = []
    
    for _ in range(Q):
        q_type = int(input_data[ptr])
        ptr += 1
        
        if q_type == 1:
            # Plant a plant. Its birth_height is the current cumulative time.
            heapq.heappush(pq, current_time)
        elif q_type == 2:
            T = int(input_data[ptr])
            ptr += 1
            current_time += T
        elif q_type == 3:
            H = int(input_data[ptr])
            ptr += 1
            
            # Condition: current_time - birth_height >= H
            # birth_height <= current_time - H
            threshold = current_time - H
            count = 0
            while pq and pq[0] <= threshold:
                heapq.heappop(pq)
                count += 1
            results.append(str(count))
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == '__main__':
    solve()