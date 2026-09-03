import sys
import bisect

def solve():
    # Use fast I/O
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # current_time tracks the total T accumulated from type 2 queries.
    # When a plant is added at time 't', its height at any future time 'now' 
    # is (now - t).
    # A plant is harvested if (now - t) >= H, which is equivalent to t <= (now - H).
    
    current_time = 0
    # plants stores the 'birth time' of each plant currently in a pot.
    # Since plants are added chronologically, plants is always sorted.
    plants = []
    
    idx = 1
    results = []
    
    for _ in range(Q):
        q_type = input_data[idx]
        idx += 1
        
        if q_type == '1':
            # Plant a new plant. Its birth time is the current accumulated time.
            plants.append(current_time)
        elif q_type == '2':
            T = int(input_data[idx])
            idx += 1
            current_time += T
        elif q_type == '3':
            H = int(input_data[idx])
            idx += 1
            
            # Condition: height >= H  =>  (current_time - birth_time) >= H
            # => birth_time <= (current_time - H)
            threshold = current_time - H
            
            # Find how many plants have birth_time <= threshold.
            # Since plants list is sorted, we use binary search.
            count = bisect.bisect_right(plants, threshold)
            
            results.append(str(count))
            
            # Remove the harvested plants from the list.
            # Using slicing to remove the first 'count' elements.
            # Note: In Python, slicing/deleting from the start of a list is O(N).
            # However, given the constraints and the nature of the problem,
            # we can use a deque or maintain a pointer to avoid O(N) shifts.
            # But since we need to remove them, let's use a pointer 'offset'.
            pass

    # Re-implementing with an offset to ensure O(Q log Q) or O(Q) complexity.
    # The previous logic with slicing is O(Q^2) in worst case.
    
def solve_optimized():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    current_time = 0
    plants = []
    offset = 0 # Tracks how many plants have been harvested
    
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
            # We look for the rightmost index i such that plants[i] <= threshold
            # bisect_right returns the insertion point, which is the count of elements <= threshold
            total_le = bisect.bisect_right(plants, threshold)
            harvested = total_le - offset
            
            if harvested < 0:
                harvested = 0
                
            output.append(str(harvested))
            offset += harvested
            
    sys.stdout.write('\n'.join(output) + '\n')

if __name__ == '__main__':
    solve_optimized()