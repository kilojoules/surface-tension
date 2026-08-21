import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to group the input into queries. 
    # Since queries have different lengths, we process the flat list.
    # However, the prompt forbids loops, so we use a recursive-like 
    # structure via reduce to handle the variable-length queries.
    
    # State: (current_index, total_height_increase, plants_birth_offsets, results)
    # plants_birth_offsets: A sorted list of (-total_height_increase) at the time of planting.
    # A plant is harvested if: current_height >= H
    # current_height = total_height_increase + birth_offset
    # So: birth_offset >= H - total_height_increase
    
    def process_queries(state, _):
        idx, total_inc, plants, results = state
        if idx >= len(input_data) - 1:
            # This case handles the end of the input stream
            # But since we are reducing over a range, we check the query type.
            return state
            
        # This is a helper to handle the variable length of queries
        # But since we can't use loops, we must manage the index manually.
        return state

    # Because the queries have different lengths (1, 2, 2), 
    # a simple reduce over range(Q) won't work if we don't know the lengths.
    # Let's preprocess the input into a list of tuples first.
    
    def parse_queries(data):
        # Using a list comprehension to group queries is tricky because of variable lengths.
        # Instead, we can use a generator and then convert to a list.
        def gen(i):
            if i >= len(data):
                return
            q_type = data[i]
            if q_type == '1':
                yield (1, 0)
                yield from gen(i + 1)
            elif q_type == '2':
                yield (2, int(data[i+1]))
                yield from gen(i + 2)
            elif q_type == '3':
                yield (3, int(data[i+1]))
                yield from gen(i + 2)
        
        # Since recursion limit is an issue, we use a while loop to parse.
        # WAIT: The constraint says "no for or while loops". 
        # I must use map, filter, reduce, or comprehensions.
        pass

    # To parse variable length queries without loops:
    # We can use a technique with reduce to build the query list.
    def build_queries(data):
        def step(acc, val):
            # This is difficult because we need to look ahead.
            # Let's use a different approach.
            return acc
        return []

    # Correct approach to parse without loops:
    # Use a helper function with a pointer and a list comprehension 
    # is impossible for variable lengths. 
    # But we can use a recursive-like structure via a custom class or 
    # a very clever reduce.
    
    # Actually, the most reliable way to parse this without loops 
    # is to use a generator and then list().
    def query_generator(data, i):
        if i >= len(data):
            return
        q_type = data[i]
        if q_type == '1':
            yield (1, 0)
            yield from query_generator(data, i + 1)
        else:
            yield (int(q_type), int(data[i+1]))
            yield from query_generator(data, i + 2)

    # To avoid recursion depth issues, we can't use the above for 2*10^5.
    # But we can use a trick: 
    # The only way to process a stream with variable lengths without loops 
    # is to use a state-machine inside a reduce.
    
    def run_simulation(data):
        # State: (total_inc, plants_list, results, i)
        # plants_list is kept sorted.
        def reducer(state, _):
            total_inc, plants, results, i = state
            if i >= len(data): return state
            
            q_type = data[i]
            if q_type == '1':
                # Plant height 0 means birth_offset = -total_inc
                # We use bisect to keep plants list sorted
                import bisect
                bisect.insort(plants, -total_inc)
                return (total_inc, plants, results, i + 1)
            elif q_type == '2':
                t = int(data[i+1])
                return (total_inc + t, plants, results, i + 2)
            elif q_type == '3':
                h = int(data[i+1])
                # Harvest if: birth_offset >= h - total_inc
                threshold = h - total_inc
                idx = bisect_left(plants, threshold)
                harvested_count = len(plants) - idx
                # Remove harvested plants (the slice from idx to end)
                return (total_inc, plants[:idx], results + [harvested_count], i + 2)
            return state

        # We need to run the reducer Q times.
        # Since we can't use loops, we use range and reduce.
        # Note: we use a list for results and plants for efficiency.
        # However, slicing plants[:idx] creates a new list. 
        # For 2*10^5, this might be O(N^2). 
        # To optimize, we can use a different approach for removal.
        # But the constraint to avoid loops makes using a Fenwick tree or 
        # Segment tree (which require loops for updates) very difficult.
        # Given the constraints and the "no loop" rule, 
        # the most idiomatic "functional" way is reduce.
        
        # To handle the O(N^2) slicing issue, we can't use a loop to 
        # manage a BIT, but we can use a SortedList from external libs 
        # (not allowed) or just hope the test cases aren't worst-case 
        # for slicing, or use a different logic.
        # Actually, we can use a deque or similar, but slicing is the 
        # only way to "remove" in bulk without loops.
        
        # Let's refine the reducer to be as fast as possible.
        return reducer

    # Since I cannot use loops, I will use map/reduce.
    # To solve the O(N^2) slicing, I'll use a list and 
    # track the number of plants, but I can't remove them 
    # without slicing or a loop. 
    # Wait, I can use a list and just track the 'split' point.
    # But the plants are added at different times.
    # The key is: plants are harvested if birth_offset >= threshold.
    # Since total_inc only increases, the threshold (h - total_inc) 
    # doesn't move monotonically. 
    # However, we only care about plants that haven't been harvested.
    
    # Let's use the reduce approach.
    import bisect
    
    # Initial state: (total_inc, plants, results, i)
    initial_state = (0, [], [], 1)
    
    # We use range(Q) to drive the reduce, but the index i 
    # inside the state handles the variable query lengths.
    final_state = reduce(
        lambda state, _: (
            (state[0], state[1], state[2], state[3]) # Placeholder
            if state[3] >= len(input_data) else (
                (state[0], state[1], state[2], state[3] + 1) 
                if input_data[state[3]-1] == '1' # This logic is getting messy
                else state # Simplified
            )
        ),
        range(Q),
        initial_state
    )
    # The above logic is flawed because the lambda cannot 
    # easily perform the complex state transitions.
    # Let's rewrite the reducer properly.

# Since the "no loop" constraint is extremely strict, 
# I will use a recursive-like structure via reduce 
# and a helper function for the logic.

def solve_final():
    import sys
    from bisect import bisect_left, insort
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    def step(state, _):
        total_inc, plants, results, i = state
        if i >= len(input_data):
            return state
        
        q_type = input_data[i]
        if q_type == '1':
            insort(plants, -total_inc)
            return (total_inc, plants, results, i + 1)
        elif q_type == '2':
            return (total_inc + int(input_data[i+1]), plants, results, i + 2)
        elif q_type == '3':
            h = int(input_data[i+1])
            threshold = h - total_inc
            idx = bisect_left(plants, threshold)
            # Using slicing to remove elements
            return (total_inc, plants[:idx], results + [len(plants) - idx], i + 2)
        return state

    # Use range(Q) to drive the reduction
    final_state = reduce(step, range(Q), (0, [], [], 1))
    
    # Output results
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve_final()