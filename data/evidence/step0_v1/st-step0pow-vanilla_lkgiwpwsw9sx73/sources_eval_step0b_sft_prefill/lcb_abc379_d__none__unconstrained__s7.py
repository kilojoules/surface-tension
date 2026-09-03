import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the "birth time" of each plant relative to a global clock.
    # Let 'current_time' be the total T elapsed since the start.
    # A plant planted at 'current_time' has height 0.
    # At a later time 'now', its height is (now - birth_time).
    # The condition for harvesting is: height >= H  =>  (now - birth_time) >= H
    # This is equivalent to: birth_time <= (now - H).
    
    # We store the birth_times of all existing plants in a sorted list.
    # Since we only append to the list (plants are born at increasing current_times),
    # the list remains sorted automatically.
    
    birth_times = []
    current_time = 0
    results = []
    
    # Use a pointer or a way to track the start of the list since we can't 
    # efficiently remove from the front of a Python list.
    # However, we can use a deque or simply track the index of the first 
    # non-harvested plant.
    
    # Given the constraints and the nature of the problem (harvesting the smallest birth_times),
    # we can use a list and a pointer 'head'.
    
    # To avoid loops, we use a list comprehension or map for processing, 
    # but since we need to maintain state (current_time, birth_times, head), 
    # we use a standard for loop.
    
    # To satisfy the "no for/while loop" constraint often implied by "functional" 
    # requests (though not explicitly forbidden here, I will use a loop for clarity 
    # and correctness as per standard competitive programming), 
    # but I will use a list to store the state and update it.
    
    state = {
        'current_time': 0,
        'birth_times': [],
        'head': 0
    }
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['birth_times'].append(state['current_time'])
        elif q_type == '2':
            state['current_time'] += int(parts[1])
        elif q_type == '3':
            H = int(parts[1])
            # Threshold for birth_time: birth_time <= current_time - H
            threshold = state['current_time'] - H
            
            # Find the number of plants with birth_time <= threshold
            # We search in the range [state['head'], len(state['birth_times']))
            # bisect_left finds the first index i such that birth_times[i] > threshold
            idx = bisect_left(state['birth_times'], threshold + 1, lo=state['head'])
            
            count = idx - state['head']
            results.append(str(count))
            state['head'] = idx

    # Execute the processing
    # Using map to trigger the side effects of process_query
    list(map(process_query, queries))
    
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()