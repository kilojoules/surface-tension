import sys
from bisect import bisect_left

def solve():
    # Read all input at once for performance
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # current_time tracks the total growth T accumulated since the start.
    # When a plant is planted at time 't', its height at time 'now' is (now - t).
    # A plant is harvested if (current_time - plant_creation_time) >= H.
    # This is equivalent to: plant_creation_time <= current_time - H.
    
    current_time = 0
    # Sorted list of creation times of existing plants.
    # Using a sorted list allows us to use binary search to find plants to harvest.
    plants = []
    
    # Pointer to track the input data
    ptr = 1
    results = []
    
    # Since we need to remove elements from the start of the list, 
    # and Python lists are slow for pop(0), we can use a deque or 
    # simply maintain a pointer to the first active plant.
    # However, since we only remove from the left, a deque or 
    # slicing/tracking an index is efficient.
    
    # Given the constraints and the nature of the problem (removing prefixes),
    # we can use a list and a start_index.
    start_idx = 0
    
    for _ in range(Q):
        q_type = input_data[ptr]
        ptr += 1
        
        if q_type == '1':
            # Plant a new plant. Its "creation time" is the current_time.
            plants.append(current_time)
        elif q_type == '2':
            T = int(input_data[ptr])
            ptr += 1
            current_time += T
        elif q_type == '3':
            H = int(input_data[ptr])
            ptr += 1
            
            # Condition: current_time - creation_time >= H
            # creation_time <= current_time - H
            threshold = current_time - H
            
            # Find how many plants have creation_time <= threshold.
            # We search in the range [start_idx, len(plants)).
            # bisect_left returns the leftmost insertion point to maintain order.
            # Since creation times are added monotonically, the list is already sorted.
            
            # We need the number of elements <= threshold.
            # bisect_right finds the index of the first element > threshold.
            import bisect
            idx = bisect.bisect_right(plants, threshold, lo=start_idx)
            
            harvested_count = idx - start_idx
            results.append(str(harvested_count))
            
            # Update start_idx to "remove" the harvested plants.
            start_idx = idx
            
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()