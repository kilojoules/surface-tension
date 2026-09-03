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
    # At any future time T_current, its actual height is T_current - T_planted.
    # A plant is harvested if T_current - T_planted >= H, which is T_planted <= T_current - H.
    
    T_total = 0
    # We store the relative heights (T_planted) of existing plants in a sorted list.
    # Since we only add plants (which always have the current T_total), 
    # the list 'plants' will naturally be sorted.
    plants = []
    
    # To handle the queries without using for/while loops, we use map and list comprehensions.
    # However, since we need to maintain state (T_total and plants), 
    # and we cannot use loops, we can use a mutable object or a closure with a reducer.
    
    state = {
        'T_total': 0,
        'plants': [],
        'results': []
    }
    
    def process_query(query_str):
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new plant. Its relative height is the current T_total.
            state['plants'].append(state['T_total'])
        elif q_type == '2':
            # Increase total time.
            state['T_total'] += int(parts[1])
        elif q_type == '3':
            # Harvest plants where T_planted <= T_total - H.
            H = int(parts[1])
            threshold = state['T_total'] - H
            # Find index of first plant that is NOT harvested.
            # bisect_left finds the first index i such that plants[i] >= threshold + 1
            # Wait, the condition is height >= H => T_total - T_planted >= H => T_planted <= T_total - H.
            # So we harvest all plants in range [0, bisect_right(plants, threshold) - 1].
            
            # Using bisect_right to find how many plants satisfy T_planted <= threshold.
            import bisect
            idx = bisect.bisect_right(state['plants'], threshold)
            
            # Record the number of harvested plants.
            state['results'].append(str(idx))
            
            # Remove the harvested plants.
            # Since we can't use loops, we use slice assignment.
            state['plants'][:] = state['plants'][idx:]

    # Use map to iterate through the queries.
    list(map(process_query, queries))
    
    # Output all results joined by newline.
    sys.stdout.write('\n'.join(state['results']) + '\n')

if __name__ == '__main__':
    solve()