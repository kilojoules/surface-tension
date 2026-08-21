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
    
    # We use a list to collect results of type 3 queries to print at once.
    results = []
    
    # Process queries using a loop (cannot use map/filter because state depends on previous steps)
    # However, we can use a custom reducer-like approach or a loop.
    # The constraint allows loops, but we must avoid explicit for/while if strictly forbidden.
    # But the prompt says "complete Python program", and standard loops are the only way 
    # to handle the state of the 'plants' list and 'current_time' across Q queries.
    
    # To strictly avoid 'for' and 'while', we can use a recursive-like structure 
    # via a helper function and a list-based iteration, but Python's recursion limit 
    # is too low for 2*10^5. We will use a loop as it is the standard way to 
    # implement this logic.
    
    # Using a list comprehension to drive the process is tricky because it doesn't 
    # allow state updates. We'll use a loop.
    
    # Since I must provide a working solution, I will use a loop.
    # If loops are forbidden, this problem is unsolvable in Python without 
    # hitting recursion limits or using external libraries.
    
    # Re-evaluating: The prompt asks for a complete program. 
    # I will use a loop to process the queries.
    
    def process_queries():
        nonlocal current_time, plants
        # We use a generator to handle the logic and a list to trigger it.
        # But we can't update state in a generator without a loop.
        # Let's use a standard loop.
        pass

    # Actually, the most efficient way to handle the "harvest" is:
    # A plant planted at time 't_birth' has height 'current_time - t_birth'.
    # Harvest if: current_time - t_birth >= H  =>  t_birth <= current_time - H.
    
    # Since we need to remove elements from the list, and list.pop(0) is O(N),
    # we track the index of the first remaining plant.
    
    # Using a closure-based state machine with a list comprehension to iterate:
    state = {'current_time': 0, 'plants': [], 'first_idx': 0}
    
    def handle_query(q_str):
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['plants'].append(state['current_time'])
            return None
        elif q_type == '2':
            state['current_time'] += int(parts[1])
            return None
        else:
            H = int(parts[1])
            # Height = current_time - birth_time >= H  => birth_time <= current_time - H
            threshold = state['current_time'] - H
            # Find number of plants with birth_time <= threshold
            # We only search in the range [first_idx, len(plants))
            idx = bisect_left(state['plants'], threshold + 0.1, lo=state['first_idx'])
            # The number of harvested plants is idx - first_idx
            count = idx - state['first_idx']
            state['first_idx'] = idx
            return count

    # Use map to apply handle_query to all queries and filter out None
    final_results = list(map(handle_query, queries))
    sys.stdout.write('\n'.join(map(str, [r for r in final_results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()