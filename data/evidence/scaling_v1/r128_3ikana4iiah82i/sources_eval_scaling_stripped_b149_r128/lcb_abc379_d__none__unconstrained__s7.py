import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_time tracks the total T elapsed since the start
    # plants stores the 'birth time' of each plant
    # A plant born at time 't' has height (current_time - t) at any later time
    # The condition height >= H becomes: current_time - t >= H  =>  t <= current_time - H
    
    current_time = 0
    plants = []
    
    # We need to process queries and collect outputs for type 3
    # Since we cannot use loops, we use a generator/map approach
    # However, the state (current_time, plants) must be updated.
    # We can use a mutable object (like a dictionary) to track state across a map.
    
    state = {
        'time': 0,
        'plants': [],
        'outputs': []
    }
    
    def process_query(q_str):
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new plant at the current time
            state['plants'].append(state['time'])
            # We must keep the plants list sorted to use bisect. 
            # Since state['time'] is non-decreasing, append maintains sort.
            return None
        
        elif q_type == '2':
            # Increase current time
            T = int(parts[1])
            state['time'] += T
            return None
        
        elif q_type == '3':
            # Harvest plants where birth_time <= current_time - H
            H = int(parts[1])
            threshold = state['time'] - H
            
            # Find number of plants with birth_time <= threshold
            # bisect_left returns the index of the first element > threshold
            idx = bisect_left(state['plants'], threshold + 1)
            
            # The number of harvested plants is the count of elements before idx
            count = idx
            
            # Remove the harvested plants from the list
            # We use slice assignment to modify the list in place
            state['plants'][0:idx] = []
            
            return count

    # Use map to apply the function to all queries
    # We use a list comprehension to trigger the map execution
    results = [process_query(q) for q in queries]
    
    # Filter out None values and print the results
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()