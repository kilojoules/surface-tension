import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the total time elapsed (T_total).
    # When a plant is planted at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_planted.
    # A plant is harvested if T_curr - T_planted >= H, which is T_planted <= T_curr - H.
    
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only append to the list (T_total only increases), 
    # the list of T_planted values will naturally be sorted.
    
    T_total = 0
    planted_times = []
    results = []
    
    # Use a pointer or slice to track which plants are still in the pots.
    # Since we remove from the beginning (smallest T_planted), 
    # we can use a deque or simply track the index of the first remaining plant.
    # However, Python's list slicing/popping from the front is O(N).
    # Given Q=2*10^5, we should avoid O(N) deletions.
    
    # Let's use a list and a pointer 'head' to simulate the queue.
    # Because we need to find how many elements are <= (T_total - H),
    # and the list is sorted, we can use bisect_left.
    
    # To handle the "removal", we can't easily remove from the middle, 
    # but we only remove from the front.
    
    # State is maintained in a dictionary or list accessed by a closure/loop.
    # Since we can't use loops, we use a mutable object to keep track of the 'head'.
    state = {
        'T_total': 0,
        'planted_times': [],
        'head': 0
    }
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['planted_times'].append(state['T_total'])
        elif q_type == '2':
            state['T_total'] += int(parts[1])
        elif q_type == '3':
            H = int(parts[1])
            threshold = state['T_total'] - H
            
            # Find index of first element > threshold
            # We only search in the range [state['head'], len(state['planted_times']))
            # bisect_left returns the index in the original list.
            idx = bisect_left(state['planted_times'], threshold + 1, lo=state['head'])
            
            # Number of plants harvested is the number of plants from 'head' to 'idx - 1'
            count = idx - state['head']
            results.append(str(count))
            
            # Update head to "remove" harvested plants
            state['head'] = idx

    # Map the process_query function across all queries
    list(map(process_query, queries))
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()