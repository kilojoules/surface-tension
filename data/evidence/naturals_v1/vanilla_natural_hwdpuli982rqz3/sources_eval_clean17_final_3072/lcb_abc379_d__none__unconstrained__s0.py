import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    Q = int(input_data[0])
    queries = input_data[1:]

    # We track the total elapsed time (T_total).
    # When a plant is planted at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_planted.
    # A plant is harvested if T_curr - T_planted >= H, which is T_planted <= T_curr - H.
    
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only add plants (which always have the current T_total as their T_planted),
    # the list remains sorted if we just append.
    
    T_total = 0
    plants = []
    results = []

    for query_str in queries:
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new plant. Its height is 0, so T_planted = T_total.
            plants.append(T_total)
        elif q_type == '2':
            # Increase height of all plants by T.
            T_total += int(parts[1])
        elif q_type == '3':
            # Harvest plants where T_total - T_planted >= H  => T_planted <= T_total - H.
            H = int(parts[1])
            threshold = T_total - H
            
            # Find how many plants have T_planted <= threshold.
            # bisect_left returns the index of the first element >= threshold + 1 (or just > threshold).
            # Since we need T_planted <= threshold, we use bisect_right or bisect_left with a logic.
            # Let's use bisect_right to find the number of elements <= threshold.
            
            # Import bisect_right inside solve or at top.
            from bisect import bisect_right
            idx = bisect_right(plants, threshold)
            
            # The number of harvested plants is idx.
            results.append(str(idx))
            
            # Remove these plants from the list. 
            # Slicing is O(N), but we can't use a loop. 
            # Given Q=2*10^5, O(N) slicing in a loop might be slow, 
            # but we only remove from the front.
            # To avoid O(N) deletes, we can use a pointer or a deque, 
            # but since we need bisect, we can't use deque.
            # However, we can just keep track of the 'start' index.
            
            # Wait, slicing is not allowed if we want true O(log N).
            # Let's use a list and a pointer to track the first non-harvested plant.
            pass

# Redefining solve to use a pointer instead of slicing to ensure efficiency.
def solve_final():
    import sys
    from bisect import bisect_right
    
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    Q = int(input_data[0])
    queries = input_data[1:]

    T_total = 0
    plants = []
    # We use a list to store T_planted. Since we only remove from the left,
    # we can track the index of the first active plant.
    # But bisect_right works on the whole list. 
    # To make it work with an offset, we can use the 'lo' argument of bisect_right.
    
    # State is kept in a dictionary or list to be accessed by the helper
    state = {'T_total': 0, 'plants': [], 'start_idx': 0}
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['plants'].append(state['T_total'])
        elif q_type == '2':
            state['T_total'] += int(parts[1])
        elif q_type == '3':
            H = int(parts[1])
            threshold = state['T_total'] - H
            # Find index of first element > threshold starting from start_idx
            idx = bisect_right(state['plants'], threshold, lo=state['start_idx'])
            count = idx - state['start_idx']
            state['start_idx'] = idx
            return str(count)
        return None

    # Use map to process and filter out None values
    output = map(process_query, queries)
    sys.stdout.write('\n'.join(filter(None, output)) + '\n')

if __name__ == "__main__":
    solve_final()