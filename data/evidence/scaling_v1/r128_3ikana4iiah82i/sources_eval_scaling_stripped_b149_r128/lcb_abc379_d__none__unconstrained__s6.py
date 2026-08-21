import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return

    Q = int(input_data[0])
    queries = input_data[1:]

    # We need to track the height of plants.
    # Let 'current_time' be the total T accumulated from type 2 queries.
    # When a plant is added at time 't', its height at any future time 'now' is (now - t).
    # A plant is harvested if (now - t) >= H, which means t <= now - H.
    
    # We will store the 'birth times' of all active plants in a sorted list.
    # Since we only add plants (type 1) and remove them (type 3), 
    # and the condition t <= now - H is a prefix of the sorted birth times,
    # we can use a sorted list and binary search.
    
    # However, Python's list.pop(0) is O(N). To keep it O(log N) or O(1) amortized,
    # we can use a deque or simply track the index of the first active plant.
    # But wait, the plants are added at different times, so the birth times 
    # are naturally added in increasing order. 
    # A simple list with a pointer (or slicing) will work.
    
    # Let's use a list to store birth times and a pointer 'start_idx' 
    # to track the first plant that hasn't been harvested.
    
    # Since we cannot use loops, we can use a generator or map.
    # But the state (current_time, plants_list, start_idx) needs to be passed.
    # We can use a mutable object (like a dictionary) to hold the state 
    # and a helper function to process each query.

    state = {
        'current_time': 0,
        'plants': [],
        'start_idx': 0
    }

    def process_query(q_str):
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new flower at the current total time
            state['plants'].append(state['current_time'])
            return None
        
        elif q_type == '2':
            # Increase total time
            state['current_time'] += int(parts[1])
            return None
        
        else: # q_type == '3'
            H = int(parts[1])
            # Condition: current_time - birth_time >= H  =>  birth_time <= current_time - H
            threshold = state['current_time'] - H
            
            # Find how many plants have birth_time <= threshold
            # We search in the range [start_idx, len(plants))
            # Use bisect_left to find the first index where birth_time > threshold
            # The number of plants harvested is (end_idx - start_idx)
            
            # We use a slice for bisect_left to avoid loops, 
            # but bisect_left works on the whole list.
            # We search for the threshold in the whole list.
            end_idx = bisect_left(state['plants'], threshold + 1)
            
            # The number of harvested plants is the number of plants 
            # from start_idx to end_idx - 1.
            # However, we must cap end_idx at the current length of plants.
            # Since we only append to plants, we don't need to worry about deletions.
            
            # We need to handle the case where the threshold is smaller than 
            # the birth time of the first active plant.
            # The number of plants harvested is max(0, end_idx - state['start_idx'])
            
            # Update the state for the next query
            # We don't actually need to remove elements from the list, 
            # just move the start_idx.
            
            # Because we are in a function called by map/generator, 
            # we must calculate the result before updating the state.
            result = max(0, end_idx - state['start_idx'])
            state['start_idx'] = state['start_idx'] + result
            return result

    # Use a generator to process queries and filter out None values
    results = filter(lambda x: x is not None, map(process_query, queries))
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()