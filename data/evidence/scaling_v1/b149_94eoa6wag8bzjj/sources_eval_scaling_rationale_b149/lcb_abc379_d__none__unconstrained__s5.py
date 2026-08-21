import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # Since we cannot use loops, we use a generator to group the input into queries.
    # However, since queries have different lengths, we'll use a custom reducer.
    
    # State structure: (current_time_offset, active_plants, results)
    # active_plants will store the 'birth time' of the plant.
    # A plant born at time 't' has height (current_time_offset - t).
    # Height >= H  =>  current_time_offset - t >= H  =>  t <= current_time_offset - H.
    
    def process_queries(state, query_tuple):
        offset, plants, results = state
        q_type = query_tuple[0]
        
        if q_type == '1':
            # Plant height 0 means its birth time is the current offset
            return (offset, plants + [offset], results)
        
        elif q_type == '2':
            # Increase height of all plants by T
            t_val = int(query_tuple[1])
            return (offset + t_val, plants, results)
        
        elif q_type == '3':
            # Harvest plants with height >= H
            h_val = int(query_tuple[1])
            # Condition: birth_time <= offset - h_val
            threshold = offset - h_val
            
            # Use list comprehensions to filter plants
            harvested_count = len([p for p in plants if p <= threshold])
            remaining_plants = [p for p in plants if p > threshold]
            
            return (offset, remaining_plants, results + [harvested_count])

    # Parsing the input into a list of tuples since we can't loop to read
    # We use a helper function to chunk the flat list based on query types
    def parse_input(data):
        # This is tricky without loops. We can use a recursive-like structure 
        # via a list comprehension if we know the structure, but the queries 
        # have variable lengths. 
        # Instead, we can map a function that tracks the index.
        # But the simplest way to handle variable length without loops 
        # is to process the flat list using a reducer that manages the pointer.
        pass

    # Revised approach: Use a reducer on the flat list of input strings.
    # State: (index, offset, plants, results)
    def flat_reducer(state, _):
        idx, offset, plants, results = state
        if idx >= len(input_data) - 1:
            return state
        
        q_type = input_data[idx + 1]
        if q_type == '1':
            # Type 1: 1 token
            return (idx + 2, offset, plants + [offset], results)
        elif q_type == '2':
            # Type 2: 2 tokens
            t_val = int(input_data[idx + 2])
            return (idx + 3, offset + t_val, plants, results)
        elif q_type == '3':
            # Type 3: 2 tokens
            h_val = int(input_data[idx + 2])
            threshold = offset - h_val
            harvested_count = len([p for p in plants if p <= threshold])
            remaining_plants = [p for p in plants if p > threshold]
            return (idx + 3, offset, remaining_plants, results + [harvested_count])

    # Since we can't use loops, we use range(Q) to drive the reducer.
    # However, the flat_reducer needs to know how many tokens to skip.
    # A better way: use a while-loop replacement via reduce on a range.
    # But the index jumps are variable. We can use a state-based reducer 
    # that consumes the input_data list.
    
    # To handle variable jumps in a reduce, the state must carry the current index.
    # We need to run the reduce Q times? No, because some queries are 1 token, some 2.
    # We can run it len(input_data) times, and the reducer just returns state if idx is past.
    
    final_state = reduce(flat_reducer, range(len(input_data)), (1, 0, [], []))
    
    # Output the results
    sys.stdout.write('\n'.join(map(str, final_state[3])) + '\n')

if __name__ == "__main__":
    solve()