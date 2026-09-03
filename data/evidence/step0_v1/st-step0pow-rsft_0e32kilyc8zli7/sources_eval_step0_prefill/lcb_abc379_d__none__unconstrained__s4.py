import sys
from bisect import bisect_left

def solve():
    # Read all input at once for speed
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # current_time tracks the total height added to all plants since the start.
    # When a plant is added at time 't', its "birth height" is -t.
    # Its actual height at time 'T' is T + (-t).
    # A plant is harvested if T + birth_height >= H, which means birth_height >= H - T.
    
    current_time = 0
    # birth_heights stores the relative birth times of plants currently in pots.
    # Since we only add plants (type 1) and remove them (type 3), 
    # and birth_heights are added in non-increasing order (because current_time increases),
    # we can use a sorted list to perform binary searches.
    # However, plants are added at different current_times. 
    # Let's store the 'birth_height' as -current_time.
    # Because current_time is non-decreasing, -current_time is non-increasing.
    # To use bisect, we need a sorted list. We can store them and sort or use a different approach.
    
    # Actually, since we need to remove elements from the middle/end, 
    # and Q is 2*10^5, a simple list with bisect and slice/pop might be O(Q^2) in worst case.
    # But wait, the condition is height >= H. 
    # Height = current_time + birth_height.
    # So we harvest plants where birth_height >= H - current_time.
    # Since birth_heights are added as -current_time, they are added in non-increasing order.
    # Example: 
    # Q1: Type 1 -> birth_height = 0, current_time = 0. List: [0]
    # Q2: Type 2 (15) -> current_time = 15.
    # Q3: Type 1 -> birth_height = -15, current_time = 15. List: [0, -15]
    # Q4: Type 3 (10) -> Harvest if birth_height >= 10 - 15 = -5.
    # Plants with birth_height 0 are harvested. List becomes [-15].
    
    # Since birth_heights are added in non-increasing order, the list is always sorted 
    # in descending order. To use bisect, we can store them as positive current_time 
    # and flip the logic, or just store them and realize that the plants 
    # most likely to be harvested are the oldest ones (the ones added earliest).
    
    # Let's store the 'time of planting' in a list.
    # Plant added at time 't' has height (current_time - t).
    # Harvest if (current_time - t) >= H  =>  t <= current_time - H.
    # The planting times 't' are added in non-decreasing order.
    # This means the list of planting times is always sorted!
    
    planting_times = []
    current_time = 0
    
    # Use a pointer or binary search to find how many plants have t <= current_time - H.
    # Since we remove plants from the beginning of the list, we can use a deque 
    # or just keep track of the offset (start index).
    
    start_idx = 0
    
    # Process queries
    ptr = 1
    results = []
    
    # We use a list and a start_idx to simulate a queue.
    # Since we only remove from the front, this is efficient.
    
    # However, the problem says we harvest ALL plants with height >= H.
    # Because planting_times is sorted, all plants from index 'start_idx' 
    # up to some index 'k' will satisfy t <= current_time - H.
    
    # To avoid O(N) deletions, we use a list and a pointer.
    # But we can't just use a pointer because we need to know how many are left.
    # Actually, we can. The number of plants harvested is (k - start_idx).
    
    # Wait, the plants are added at the end and removed from the front.
    # This is exactly a queue.
    
    # Let's refine:
    # Type 1: planting_times.append(current_time)
    # Type 2: current_time += T
    # Type 3: 
    #   threshold = current_time - H
    #   Find number of elements in planting_times[start_idx:] that are <= threshold.
    #   Since planting_times is sorted, we find the upper bound of 'threshold'.
    #   The number of plants is (idx - start_idx).
    #   Update start_idx = idx.
    
    # Using a list and bisect_right on the slice is slow. 
    # We should bisect on the whole list and handle the start_idx.
    
    # Using sys.stdin.read().split() and a loop:
    it = iter(input_data)
    Q = int(next(it))
    
    planting_times = []
    current_time = 0
    start_idx = 0
    
    output = []
    
    for _ in range(Q):
        q_type = next(it)
        if q_type == '1':
            planting_times.append(current_time)
        elif q_type == '2':
            current_time += int(next(it))
        else:
            H = int(next(it))
            threshold = current_time - H
            
            # Find the number of plants with planting_time <= threshold
            # We only care about plants from start_idx onwards.
            # bisect_right returns the insertion point to maintain order.
            import bisect
            idx = bisect.bisect_right(planting_times, threshold, lo=start_idx)
            
            count = idx - start_idx
            output.append(str(count))
            start_idx = idx
            
    sys.stdout.write('\n'.join(output) + '\n')

if __name__ == '__main__':
    solve()