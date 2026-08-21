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
    # When a plant is created at T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - T_total.
    # The condition height >= H becomes: T_curr - T_total >= H  => T_total <= T_curr - H.
    
    t_total = 0
    # Sorted list of T_total values for all existing plants
    plants = []
    
    # Since we cannot use loops, we use map/list comprehensions and 
    # a mutable state captured in a dictionary or list.
    state = {
        't_total': 0,
        'plants': [],
        'results': []
    }
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new flower. Its relative height is -state['t_total']
            # We store state['t_total'] to check against (state['t_total'] - H)
            # Actually, let's store the time it was planted.
            # Height = Current_Total_T - Planted_T.
            # Harvest if Current_Total_T - Planted_T >= H  => Planted_T <= Current_Total_T - H.
            state['plants'].append(state['t_total'])
            # We need the list to stay sorted to use bisect. 
            # Since state['t_total'] only increases, appending keeps it sorted.
            
        elif q_type == '2':
            t_inc = int(parts[1])
            state['t_total'] += t_inc
            
        elif q_type == '3':
            h = int(parts[1])
            threshold = state['t_total'] - h
            
            # Find number of plants with Planted_T <= threshold
            # bisect_right returns the index where threshold would be inserted while maintaining order
            # Since we can't use loops, we use bisect_right from the bisect module.
            from bisect import bisect_right
            idx = bisect_right(state['plants'], threshold)
            
            # The number of harvested plants is idx.
            state['results'].append(str(idx))
            
            # Remove harvested plants. 
            # Since we can't use loops, we use slice assignment.
            state['plants'] = state['plants'][idx:]

    # Use map to apply the process_query function to all queries
    list(map(process_query, queries))
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(state['results']) + '\n')

if __name__ == "__main__":
    solve()