import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]

    # State structure: (current_total_height, sorted_list_of_birth_offsets, output_list)
    # A plant's current height is: current_total_height - birth_offset
    # Condition height >= H  =>  current_total_height - birth_offset >= H
    #                         =>  birth_offset <= current_total_height - H
    
    def process_query(state, query_str):
        total_h, plants, outputs = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant height 0 means its birth_offset is the current total_h
            # We maintain the plants list sorted to use binary search
            # Since total_h is non-decreasing, we can just append
            plants.append(total_h)
            return (total_h, plants, outputs)
        
        elif q_type == 2:
            # Increase total height
            return (total_h + parts[1], plants, outputs)
        
        else: # q_type == 3
            H = parts[1]
            # Threshold for birth_offset: birth_offset <= total_h - H
            threshold = total_h - H
            # Find how many plants have birth_offset <= threshold
            idx = bisect_left(plants, threshold + 1)
            
            # The number of harvested plants is idx
            # We remove the first 'idx' elements from the list
            # Note: slicing creates a new list, which is acceptable given constraints
            # but we must be careful with performance. 
            # However, since we can't use loops, we rely on Python's optimized slicing.
            outputs.append(str(idx))
            return (total_h, plants[idx:], outputs)

    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(process_query, queries, (0, [], []))
    
    # Print all harvested counts joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()