import sys
from bisect import bisect_left

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # current_time tracks the total T accumulated from type 2 queries.
    # When a plant is planted at time 't', its height at time 'now' is (now - t).
    # A plant is harvested if (current_time - plant_creation_time) >= H.
    # This is equivalent to: plant_creation_time <= current_time - H.
    
    current_time = 0
    # plants stores the 'creation_time' of each plant currently in a pot.
    # Since plants are added chronologically, this list is naturally sorted.
    plants = []
    
    idx = 1
    results = []
    
    for _ in range(Q):
        q_type = input_data[idx]
        idx += 1
        
        if q_type == '1':
            # Plant a new plant. Its "birth time" is the current accumulated T.
            plants.append(current_time)
        elif q_type == '2':
            T = int(input_data[idx])
            idx += 1
            current_time += T
        elif q_type == '3':
            H = int(input_data[idx])
            idx += 1
            
            # Condition for harvest: current_time - birth_time >= H
            # birth_time <= current_time - H
            threshold = current_time - H
            
            # Find how many plants have birth_time <= threshold.
            # Since plants list is sorted, we use binary search.
            count = bisect_left(plants, threshold + 1) 
            # Note: bisect_left(plants, threshold + 1) gives the index of the first 
            # element > threshold, which is exactly the number of elements <= threshold.
            
            results.append(str(count))
            
            # Remove the harvested plants from the list.
            # Using slicing to remove the first 'count' elements.
            # While slicing creates a new list, given the constraints and the 
            # nature of the problem, it is generally acceptable in Python 
            # for 2*10^5 elements, though a deque or pointer would be O(1).
            # However, since we remove from the front, we can just maintain a pointer.
            # Let's use a pointer 'offset' to avoid O(N) deletions.
            pass

    # To avoid the O(N) cost of list slicing, I will rewrite the loop using an offset.

def solve_optimized():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    current_time = 0
    plants = []
    offset = 0 # Tracks the start of the active plants list
    
    idx = 1
    output = []
    
    for _ in range(Q):
        q_type = input_data[idx]
        idx += 1
        if q_type == '1':
            plants.append(current_time)
        elif q_type == '2':
            T = int(input_data[idx])
            idx += 1
            current_time += T
        else:
            H = int(input_data[idx])
            idx += 1
            threshold = current_time - H
            
            # Binary search in the range [offset, len(plants))
            # We want the number of plants with birth_time <= threshold
            # search for the first index i where plants[i] > threshold
            
            # Adjust bisect to search within the slice [offset:]
            # We search the whole list and then subtract the offset.
            # Since the list is sorted, we find the first index i such that plants[i] > threshold.
            
            # Using a custom binary search to handle the offset
            low = offset
            high = len(plants)
            while low < high:
                mid = (low + high) // 2
                if plants[mid] <= threshold:
                    low = mid + 1
                else:
                    high = mid
            
            harvested_count = low - offset
            output.append(str(harvested_count))
            offset = low
            
    sys.stdout.write('\n'.join(output) + '\n')

if __name__ == '__main__':
    solve_optimized()