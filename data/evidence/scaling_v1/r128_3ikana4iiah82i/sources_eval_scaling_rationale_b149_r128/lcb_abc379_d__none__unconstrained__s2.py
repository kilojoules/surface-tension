import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to parse the queries. Since they have different lengths, 
    # we process the flat list using a custom reducer or a generator.
    # However, the constraint forbids 'for' or 'while' loops.
    # We can use a recursive-like structure via reduce to handle the variable-length queries.
    
    # State: (current_time, plants_birth_times, results, input_ptr)
    # plants_birth_times is a sorted list of times when plants were added.
    # A plant added at time 't' has height (current_time - t).
    # It is harvested if (current_time - t) >= H  =>  t <= (current_time - H).
    
    def process_queries(state, _):
        current_time, plants, results, ptr = state
        q_type = input_data[ptr]
        
        if q_type == '1':
            # Plant added at current_time
            return (current_time, plants + [current_time], results, ptr + 1)
        
        elif q_type == '2':
            # Wait T days
            T = int(input_data[ptr + 1])
            return (current_time + T, plants, results, ptr + 2)
        
        else: # q_type == '3'
            # Harvest plants with height >= H
            H = int(input_data[ptr + 1])
            # Condition: birth_time <= current_time - H
            threshold = current_time - H
            # Find number of plants with birth_time <= threshold
            # Since plants are added chronologically, the list is already sorted.
            idx = bisect_left(plants, threshold + 0.1) # Find first index > threshold
            # Actually, bisect_right is better for <= threshold
            # But we can use bisect_left with a tiny offset or just use the logic:
            # All plants from 0 to idx-1 are harvested.
            
            # Correcting the index logic:
            # We want count of t in plants such that t <= current_time - H.
            # Using a helper to find the count:
            import bisect
            count = bisect.bisect_right(plants, threshold)
            
            # Remove the first 'count' plants and record the result
            return (current_time, plants[count:], results + [str(count)], ptr + 2)

    # Using reduce to simulate the loop through the queries.
    # We run it Q times, but the ptr handles the actual position in input_data.
    final_state = reduce(process_queries, range(Q), (0, [], [], 1))
    
    # Output the results
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()