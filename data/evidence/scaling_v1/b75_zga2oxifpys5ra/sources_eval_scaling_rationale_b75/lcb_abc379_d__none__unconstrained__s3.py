import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # Since we cannot use loops, we use a generator to group the input into queries.
    def get_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, None)
                elif q_type == '2':
                    yield (2, int(next(it)))
                elif q_type == '3':
                    yield (3, int(next(it)))
            except StopIteration:
                break

    # State structure: (current_total_height, sorted_list_of_birth_offsets, results_list)
    # A plant planted at total_height 'S' has height (current_total_height - S)
    # Harvest condition: current_total_height - S >= H  =>  S <= current_total_height - H
    
    def process_query(state, query):
        total_h, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant height 0 means its birth offset is the current total height
            # We use a list and maintain it sorted. Since we only add the current 
            # total_h, and total_h is non-decreasing, the list remains sorted.
            return (total_h, plants + [total_h], results)
        
        elif q_type == 2:
            # Increase total height offset
            return (total_h + val, plants, results)
        
        else: # q_type == 3
            # Harvest plants where birth_offset <= total_h - val
            threshold = total_h - val
            # Find index of first plant with offset > threshold
            idx = bisect_left(plants, threshold + 1) 
            # Actually, bisect_left on threshold + 1 finds the first element > threshold
            # Wait, the condition is S <= total_h - H.
            # bisect_right finds the index where elements to the left are <= threshold.
            import bisect
            idx = bisect.bisect_right(plants, threshold)
            
            harvested_count = idx
            # Remove harvested plants by slicing
            return (total_h, plants[idx:], results + [harvested_count])

    # Using reduce to simulate the loop over queries
    # Initial state: (total_height, plants_list, results_list)
    final_state = reduce(process_query, get_queries(input_data), (0, [], []))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()