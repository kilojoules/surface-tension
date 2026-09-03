import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the total elapsed time (T_total).
    # When a plant is planted at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_planted.
    # A plant is harvested if T_curr - T_planted >= H, which is T_planted <= T_curr - H.
    
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only append to the list (T_total only increases), 
    # the list of T_planted will naturally be sorted.
    
    T_total = 0
    planted_times = []
    results = []
    
    # Use a pointer or slicing to handle the "removal" of harvested plants.
    # Since we can't use loops, we use a list and track the index of the first non-harvested plant.
    # However, since we need to remove elements from the front, and lists are slow at the front,
    # we can use a deque or simply track the offset. 
    # But wait, the condition is T_planted <= T_curr - H.
    # Since T_planted is monotonically increasing, the plants to be harvested 
    # are always a prefix of the current list of plants.
    
    # To avoid loops and recursion, we use a state object to keep track of the offset.
    # But we can just use a list and bisect, then slice. 
    # Slicing is O(K), but we can't do it in a loop.
    # Let's use a different approach: store all T_planted in a list.
    # When query 3 H comes, find how many elements are <= T_total - H.
    # The tricky part is that these elements must not have been harvested yet.
    
    # Let's use a list to store T_planted and an integer to store how many have been harvested.
    # Since we can't use loops, we'll use a mutable object to keep track of the state.
    
    state = {
        'T_total': 0,
        'planted_times': [],
        'harvested_count': 0
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
            idx = bisect_left(state['planted_times'], threshold + 1)
            # The number of plants to harvest is (idx - state['harvested_count'])
            # But only if idx > state['harvested_count']
            count = max(0, idx - state['harvested_count'])
            results.append(str(count))
            state['harvested_count'] = idx

    # Map the process_query function over all queries
    list(map(process_query, queries))
    
    # Print all results joined by newline
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()