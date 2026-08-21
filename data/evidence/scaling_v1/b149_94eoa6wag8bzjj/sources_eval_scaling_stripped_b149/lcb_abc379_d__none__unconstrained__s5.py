import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_height_offset tracks the total T added by type 2 queries.
    # When a plant is added at time t, its "birth height" is -current_height_offset.
    # Its actual height at any time is birth_height + current_height_offset.
    # A plant is harvested if birth_height + current_height_offset >= H,
    # which is equivalent to birth_height >= H - current_height_offset.
    
    current_height_offset = 0
    # sorted_birth_heights stores the birth heights of all active plants.
    sorted_birth_heights = []
    # We use a list to store results to print them all at once.
    results = []
    
    # Since we cannot use loops, we use a generator or map to process queries.
    # However, we need to maintain state (offset and the list of plants).
    # We can use a mutable object (like a dictionary or a class) to hold the state
    # and a helper function to process each query.
    
    state = {
        'offset': 0,
        'plants': [],
        'out': []
    }
    
    def process_query(q_str):
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant height 0 means birth_height = -current_offset
            # We use bisect.insort to keep the list sorted for binary search
            import bisect
            bisect.insort(state['plants'], -state['offset'])
        
        elif q_type == '2':
            state['offset'] += int(parts[1])
            
        elif q_type == '3':
            H = int(parts[1])
            # Condition: birth_height >= H - state['offset']
            threshold = H - state['offset']
            idx = bisect_left(state['plants'], threshold)
            
            # Number of plants to harvest
            harvested_count = len(state['plants']) - idx
            state['out'].append(str(harvested_count))
            
            # Remove harvested plants (those from idx to the end)
            # We update the list by slicing
            state['plants'] = state['plants'][:idx]

    # Use a list comprehension to trigger the side-effect of process_query for each query
    [process_query(q) for q in queries]
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(state['out']) + '\n')

if __name__ == "__main__":
    solve()