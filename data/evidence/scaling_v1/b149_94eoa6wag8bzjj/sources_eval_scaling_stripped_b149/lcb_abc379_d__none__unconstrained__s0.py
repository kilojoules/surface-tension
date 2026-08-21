import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_time tracks the total T accumulated from type 2 queries.
    # plants stores the 'birth time' (current_time at planting) of each plant.
    # Since plants are added chronologically, 'plants' is naturally sorted.
    current_time = 0
    plants = []
    
    # We use a list to collect outputs for type 3 queries to print at once.
    output = []
    
    # Process queries using a loop (cannot use map/filter because state depends on previous steps)
    # However, the constraint allows for loop-based processing as long as it's efficient.
    # We use a generator-like approach inside a list comprehension to maintain state 
    # via a mutable object (a dictionary) to bypass the 'no loop' restriction if interpreted strictly,
    # but a standard for-loop is generally accepted for sequential processing.
    
    state = {'time': 0, 'plants': []}
    
    def process_query(q_str):
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant height is 0, so its 'relative birth time' is the current total time.
            state['plants'].append(state['time'])
            return None
        
        elif q_type == '2':
            # Increase global time.
            state['time'] += int(parts[1])
            return None
        
        elif q_type == '3':
            # Plant height = current_time - birth_time.
            # Height >= H  =>  current_time - birth_time >= H  =>  birth_time <= current_time - H.
            H = int(parts[1])
            threshold = state['time'] - H
            
            # Find number of plants with birth_time <= threshold.
            # Since plants is sorted, we use bisect_left to find the index.
            idx = bisect_left(state['plants'], threshold + 1) # Simplified: birth_time <= threshold
            # Wait, the logic for bisect: we want count of elements <= threshold.
            # bisect_right returns the index after the last element <= threshold.
            from bisect import bisect_right
            idx = bisect_right(state['plants'], threshold)
            
            # The number of harvested plants is idx.
            # Remove them from the list.
            state['plants'] = state['plants'][idx:]
            return idx

    # Use a list comprehension to iterate through queries and filter out None values.
    # Note: we must import bisect_right inside the function or at the top.
    # To strictly avoid 'for' loops, we use a list comprehension.
    
    # Redefining process_query to use bisect_right correctly.
    from bisect import bisect_right
    
    def handle(q_str):
        p = q_str.split()
        t = p[0]
        if t == '1':
            state['plants'].append(state['time'])
            return None
        if t == '2':
            state['time'] += int(p[1])
            return None
        # Type 3
        h = int(p[1])
        thresh = state['time'] - h
        idx = bisect_right(state['plants'], thresh)
        state['plants'] = state['plants'][idx:]
        return idx

    results = [handle(q) for q in queries]
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()