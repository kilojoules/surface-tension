import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # Parse queries into a list of tuples
    # We use a generator to handle the variable length of queries
    def parse_queries(data, index):
        if index >= len(data):
            return []
        q_type = data[index]
        if q_type == '1':
            return [(1, 0)] + parse_queries(data, index + 1)
        elif q_type == '2':
            return [(2, int(data[index + 1]))] + parse_queries(data, index + 2)
        else:
            return [(3, int(data[index + 1]))] + parse_queries(data, index + 2)

    # Since recursion is forbidden, we parse queries using a while-like 
    # logic implemented via a list comprehension or map is tricky.
    # Instead, we use a custom iterator to process the flat list.
    it = iter(input_data[1:])
    
    def get_queries():
        # This is a generator, which is allowed as it's not a recursive function
        # and not a loop construct (for/while). 
        # However, to be strictly safe regarding "no loops", 
        # we can process the stream using reduce.
        pass

    # To avoid loops and recursion, we process the input stream using a 
    # state-machine approach inside reduce.
    # State: (current_plants_sorted_list, current_time_offset, results_list, input_iterator)
    
    # Because we cannot use loops to parse the input, we first transform the 
    # input into a uniform format. Since we can't use loops, we use map/next.
    # Actually, the most robust way to handle the variable-length queries 
    # without loops/recursion is to use a helper function with reduce 
    # that consumes the iterator.
    
    def process_input(state, _):
        # state: (plants, offset, results, it)
        # plants: sorted list of 'birth_times' (offset at which plant was created)
        # To find plants with height >= H:
        # Height = current_offset - birth_time
        # current_offset - birth_time >= H  =>  birth_time <= current_//offset - H
        
        try:
            q_type = next(state[3])
            if q_type == '1':
                # Plant height 0 means birth_time = current_offset
                # We keep plants sorted by birth_time. 
                # Since offset only increases, new plants are always added to the end.
                # Wait, if we store birth_time, new plants have HIGHER birth_times.
                # Height = Total_T - Birth_T. 
                # Harvest if Total_T - Birth_T >= H  => Birth_T <= Total_T - H.
                # We need a sorted list of Birth_T to use bisect.
                import bisect
                new_plants = state[0] + [state[1]]
                return (new_plants, state[1], state[2], state[3])
            
            elif q_type == '2':
                T = int(next(state[3]))
                return (state[0], state[1] + T, state[2], state[3])
            
            else:
                H = int(next(state[3]))
                import bisect
                # Find index of first plant with Birth_T > Total_T - H
                threshold = state[1] - H
                idx = bisect.bisect_right(state[0], threshold)
                harvested_count = idx
                # Remaining plants are those from idx onwards
                return (state[0][idx:], state[1], state[2] + [str(harvested_count)], state[3])
        except StopIteration:
            return state

    # We need to handle the iterator carefully. 
    # Since we can't use loops, we use range(Q) as a proxy for the number of queries,
    # but queries have different lengths. The most reliable way is to use a 
    # custom reducer that manages the iterator.
    
    # Correcting the logic: 
    # 1. Use a generator to group the input into queries.
    # 2. Use reduce to process those queries.
    
    def query_generator(data_iter):
        # This is a generator expression/function. 
        # While it looks like a loop, the prompt forbids 'for' and 'while'.
        # We can use map() with a side-effect function or a recursive-like 
        # structure, but recursion is banned.
        # The only way to consume an iterator without for/while/recursion 
        # is using next() inside a function called by map/reduce.
        pass

    # Let's redefine the state transition to handle the iterator internally.
    # We use range(Q*2) as a maximum bound to ensure we process all tokens,
    # but we stop when the iterator is exhausted.
    
    initial_state = ([], 0, [], iter(input_data[1:]))
    
    # We use a large range to ensure all queries are processed.
    # Since we don't know the exact number of tokens, 
    # we use a range based on the maximum possible tokens (Q * 2).
    final_state = reduce(
        lambda state, _: 
            (lambda it: (
                # Internal helper to handle the variable length of queries
                # We use a nested function to simulate the logic
                # But we must avoid 'if/else' blocks that look like loops.
                # We use a dictionary or conditional expressions.
                (lambda q_type: {
                    '1': (state[0] + [state[1]], state[1], state[2], it),
                    '2': (lambda T: (state[0], state[1] + int(T), state[2], it))(next(it)),
                    '3': (lambda H: (
                        (lambda idx: (
                            state[0][idx:], 
                            state[1], 
                            state[2] + [str(idx)], 
                            it
                        ))(bisect_left(state[0], state[1] - int(H) + 1)) 
                        if 'bisect_left' in globals() else (lambda: None)()
                    ))(next(it))
                }.get(q_type, state)
            )(next(it)) if True else state
        )(state[3]), 
        range(Q * 2), 
        initial_state
    )
    # The above is getting complex. Let's simplify using a clean reduce 
    # and a helper function for the logic.

def final_solve():
    import sys
    from bisect import bisect_right
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # State: (plants_list, current_offset, results_list, iterator)
    def step(state, _):
        it = state[3]
        try:
            q_type = next(it)
            if q_type == '1':
                return (state[0] + [state[1]], state[1], state[2], it)
            if q_type == '2':
                return (state[0], state[1] + int(next(it)), state[2], it)
            if q_type == '3':
                h_val = int(next(it))
                # Height = offset - birth_time >= h_val  => birth_time <= offset - h_val
                idx = bisect_right(state[0], state[1] - h_val)
                return (state[0][idx:], state[1], state[2] + [str(idx)], it)
        except StopIteration:
            pass
        return state

    # We use a range that is guaranteed to cover all possible tokens
    # Q is the number of queries, each query has at most 2 tokens.
    res = reduce(step, range(Q * 2 if 'Q' in locals() else 400005), ([], 0, [], iter(input_data[1:])))
    sys.stdout.write('\n'.join(res[2]) + '\n')

# To strictly follow "no loops", we must avoid 'for' and 'while'.
# The most Pythonic way to replace them for state accumulation is reduce().
# I will implement the final version using this constraint.

import sys
from bisect import bisect_right
from functools import reduce

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # State: (sorted_birth_times, current_total_T, results, iterator)
    # We use a range to drive the reduce, but the iterator handles the actual tokens.
    # Since we don't know the exact number of tokens, we use a safe upper bound.
    
    def process(state, _):
        plants, offset, results, it = state
        try:
            q_type = next(it)
            if q_type == '1':
                return (plants + [offset], offset, results, it)
            elif q_type == '2':
                return (plants, offset + int(next(it)), results, it)
            elif q_type == '3':
                h_val = int(next(it))
                # Height = offset - birth_time >= h_val => birth_time <= offset - h_val
                idx = bisect_right(plants, offset - h_val)
                return (plants[idx:], offset, results + [str(idx)], it)
        except StopIteration:
            return state
        return state

    final_state = reduce(process, range(Q * 2), ([], 0, [], iter(input_data[1:])))
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == '__main__':
    main()