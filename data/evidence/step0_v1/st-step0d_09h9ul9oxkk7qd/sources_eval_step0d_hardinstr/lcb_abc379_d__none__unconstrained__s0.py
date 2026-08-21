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

    # State structure: (current_total_height, sorted_list_of_birth_offsets, results_list)
    # A plant planted when total height was 'S' has current height: current_total_height - S
    # Condition: current_total_height - S >= H  =>  S <= current_total_height - H
    
    def process_query(state, query_str):
        current_h, plants, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant a new plant. Its "birth offset" is the current total height.
            # We use a sorted list to keep track of offsets.
            # Since we only add plants at the 'current' time, and current_h increases,
            # the offsets are naturally added in non-decreasing order.
            plants.append(current_h)
            return (current_h, plants, results)
        
        elif q_type == 2:
            # Increase total height
            return (current_h + parts[1], plants, results)
        
        else:
            # Harvest plants where height >= H
            # Height = current_h - offset >= H  => offset <= current_h - H
            h_threshold = parts[1]
            max_offset = current_h - h_threshold
            
            # Find how many plants have offset <= max_offset
            # Since plants list is sorted, we use bisect_left
            idx = bisect_left(plants, max_offset + 1)
            
            # The number of harvested plants is idx
            # We remove the first 'idx' elements from the list
            # To avoid loops and maintain efficiency, we slice the list
            # Note: slicing creates a new list, which is acceptable given Q=2e5
            # but we must be careful with memory. However, it's the only way without loops.
            results.append(str(idx))
            return (current_h, plants[idx:], results)

    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(process_query, queries, (0, [], []))
    
    # Output all results joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()