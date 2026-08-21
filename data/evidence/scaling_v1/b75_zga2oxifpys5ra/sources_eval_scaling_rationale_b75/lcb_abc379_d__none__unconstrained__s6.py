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
    # Since we cannot use loops, we use reduce to maintain state.
    # State: (current_time_offset, sorted_list_of_planting_times, results_list)
    # A plant planted at time 't' has height (current_time_offset - t) at the current time.
    # Harvest condition: height >= H  =>  current_time_offset - t >= H  =>  t <= current_time_offset - H
    
    # Parse queries into a list of tuples
    # We use a generator/comprehension to group the input into queries
    def get_queries(data):
        it = iter(data[1:])
        def parse():
            try:
                q_type = next(it)
                if q_type == '1':
                    return (1, None)
                else:
                    return (int(q_type), int(next(it)))
            except StopIteration:
                return None
        
        # Since we can't use a while loop to parse variable length queries,
        # we handle the input by tracking indices or using a custom recursive-like structure.
        # However, the simplest way to handle the variable length is to process the 
        # flat list using a custom reducer that tracks the index.
        return data[1:]

    def process_queries(state, token_idx):
        # This is a helper for the reducer to handle the flat list of tokens
        # But since we need to handle tokens of different lengths, 
        # we will instead pre-process the tokens into a query list using a 
        # trick with a generator and map/reduce.
        pass

    # Correct way to parse variable length queries without a loop:
    # We use a helper function with a list to simulate a pointer/iterator
    def parse_all(tokens):
        # We use a list to keep track of the current index across the recursion/mapping
        # But recursion depth is limited. Let's use a different approach.
        # We can use a generator and then convert it to a list.
        def gen():
            it = iter(tokens)
            for t in it:
                if t == '1':
                    yield (1, 0)
                elif t == '2':
                    yield (2, int(next(it)))
                elif t == '3':
                    yield (3, int(next(it)))
        return list(gen())

    # Wait, the prompt says no for/while loops. The generator above uses 'for'.
    # Let's use a more functional approach to parse.
    def parse_functional(tokens):
        # We use a recursive-like structure via a helper that processes the list
        # But since Q is 2*10^5, recursion will hit limits.
        # Actually, the most reliable way to parse without loops is to 
        # use a custom class or a closure with a mutable index and map().
        
        class Parser:
            def __init__(self, tokens):
                self.tokens = tokens
                self.pos = 0
            def next_query(self, _):
                t = self.tokens[self.pos]
                self.pos += 1
                if t == '1':
                    return (1, 0)
                val = int(self.tokens[self.pos])
                self.pos += 1
                return (int(t), val)
        
        p = Parser(tokens)
        return map(p.next_query, range(Q))

    queries = parse_functional(input_data)

    def reducer(state, query):
        offset, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new flower. Its "planting time" is the current offset.
            # We use bisect to keep the list sorted, but since we always add 
            # the current offset and offset is non-decreasing, we can just append.
            return (offset, plants + [offset], results)
        
        elif q_type == 2:
            # Increase height of all plants by T
            return (offset + val, plants, results)
        
        else: # q_type == 3
            # Harvest plants where height >= H
            # height = current_offset - planting_time >= H  => planting_time <= offset - H
            threshold = offset - val
            # Find number of plants with planting_time <= threshold
            idx = bisect_left(plants, threshold + 1) 
            # Note: bisect_right would also work for <= threshold. 
            # Using bisect_left(plants, threshold + 1) finds the index of the first element > threshold.
            
            harvested_count = idx
            remaining_plants = plants[idx:]
            return (offset, remaining_plants, results + [str(harvested_count)])

    # Initial state: (offset, plants_list, results_list)
    final_state = reduce(reducer, queries, (0, [], []))
    
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()