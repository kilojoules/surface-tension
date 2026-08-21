import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the "birth time" of plants relative to a global timer.
    # Let 'current_time' be the sum of all T from type 2 queries.
    # A plant planted at 'current_time' has an initial height of 0.
    # At a later time 'now', its height is: now - birth_time.
    # Condition for harvest: height >= H  =>  now - birth_time >= H  =>  birth_time <= now - H.
    
    # Since we cannot use loops, we process the queries using a list comprehension
    # and a mutable state object (a list or dictionary) to track the plants.
    # However, we need to remove plants. A sorted list of birth times allows 
    # binary search to find the range of plants to harvest.
    
    # State: [current_time, sorted_birth_times_list]
    state = [0, []]
    
    # To avoid loops, we use a helper function and map/list comprehension.
    # But we need to update the state. We can use a list to store the results.
    # Since we need to remove elements from the list, and list slicing creates new lists,
    # we will maintain the sorted list of birth times.
    
    def process_query(q_data):
        q_type = q_data[0]
        
        if q_type == '1':
            # Plant a new plant at the current global time
            state[1].append(state[0])
            # Note: state[1] remains sorted because state[0] is non-decreasing
            return None
        
        elif q_type == '2':
            # Increase global time
            t_val = int(q_data[1])
            state[0] += t_val
            return None
        
        elif q_type == '3':
            # Harvest plants where birth_time <= current_time - H
            h_val = int(q_data[1])
            threshold = state[0] - h_val
            
            # Find index of first plant that is NOT harvestable
            # Plants in range [0, idx-1] are harvested
            idx = bisect_left(state[1], threshold + 0.1) 
            # Using a small offset or logic to handle <= threshold
            # Actually, bisect_right is better for birth_time <= threshold
            import bisect
            idx = bisect.bisect_right(state[1], threshold)
            
            count = idx
            # Remove harvested plants by slicing the list
            state[1] = state[1][idx:]
            return count

    # Group the input into queries. 
    # Since queries have different lengths, we can't use a simple split.
    # We use a generator to yield queries based on the type.
    def query_generator():
        it = iter(input_data[1:])
        for item in it:
            if item == '1':
                yield ('1',)
            elif item == '2':
                yield ('2', next(it))
            elif item == '3':
                yield ('3', next(it))

    # Execute the process and filter out None values
    results = [process_query(q) for q in query_generator()]
    
    # Print the non-None results
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()