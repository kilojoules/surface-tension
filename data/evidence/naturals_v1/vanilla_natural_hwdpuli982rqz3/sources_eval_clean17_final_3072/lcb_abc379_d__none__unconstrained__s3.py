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
    # Since we only add plants (which always have the current T_total as their T_planted),
    # the list remains sorted if we just append.
    
    T_total = 0
    plants = []
    results = []
    
    # To handle the removal of elements from the middle/front of a list efficiently,
    # we can't use list.pop(0) or del. 
    # However, since we only remove plants from the "smallest T_planted" end 
    # (those planted earliest), we can maintain a pointer to the first active plant.
    
    # Given the constraints and the nature of the problem, we can use a 
    # combination of a list and an index pointer.
    
    # We need to process queries and store them because we can't use loops.
    # We use a mutable object to keep track of state across map operations.
    state = {
        'T_total': 0,
        'plants': [],
        'pointer': 0
    }
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['plants'].append(state['T_total'])
        elif q_type == '2':
            state['T_total'] += int(parts[1])
        elif q_type == '3':
            H = int(parts[1])
            # Harvest condition: T_planted <= T_total - H
            threshold = state['T_total'] - H
            
            # Find how many plants have T_planted <= threshold
            # bisect_left finds the index of the first element >= threshold + 1
            # Since the list is sorted, all elements from 'pointer' to idx-1 are harvested.
            
            # We use bisect_left on the slice or the whole list.
            # To avoid slicing, we search the whole list and subtract the current pointer.
            import bisect
            idx = bisect.bisect_right(state['plants'], threshold)
            
            count = idx - state['pointer']
            results.append(str(count))
            state['pointer'] = idx

    # Use map to iterate through the queries
    list(map(process_query, queries))
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(results) + '\n')

if __name__ == "__main__":
    solve()