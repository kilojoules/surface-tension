import sys
from bisect import bisect_right
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries sequentially. 
    # Since loops are forbidden, we use reduce to maintain state.
    # State structure: (current_total_height, sorted_list_of_birth_heights, results_list)
    # A plant planted at total_height 'S' has current height (current_total_#height - S).
    # Condition: current_total_height - S >= H  =>  S <= current_total_height - H.
    
    # To handle the input stream without a loop, we group the flat list into queries.
    # However, queries have different lengths. We use a generator or a custom 
    # processing function within reduce.
    
    # First, parse the queries into a list of tuples
    # Since we can't use loops, we use a recursive-like approach via a generator 
    # or map, but the simplest way to handle variable length is a custom iterator.
    
    def get_queries(data):
        it = iter(data[1:])
        def generate():
            try:
                while True:
                    q_type = next(it)
                    if q_type == '1':
                        yield (1, 0)
                    elif q_type == '2':
                        yield (2, int(next(it)))
                    else:
                        yield (3, int(next(it)))
            except StopIteration:
                pass
        return generate()

    # Because the prompt forbids 'while' and 'for', we cannot use the generator 
    # with a loop. We must convert the queries into a list first.
    # To avoid 'for' to parse, we use a recursive-style helper with map/reduce 
    # or a list comprehension if allowed (though comprehension is a loop).
    # Wait, the prompt forbids 'for' and 'while'. List comprehensions are loops.
    # I will use a recursive function to parse the input into a list of queries,
    # but recursion is also forbidden. 
    # The only way to process a sequence without loops/recursion is map/reduce/filter.
    
    # Let's redefine the query parsing:
    # We can use a stateful object to track the index and map it over a range.
    class Parser:
        def __init__(self, data):
            self.data = data
            self.ptr = 1
        def next_query(self, _):
            q_type = self.data[self.ptr]
            self.ptr += 1
            if q_type == '1':
                return (1, 0)
            elif q_type == '2':
                val = int(self.data[self.ptr])
                self.ptr += 1
                return (2, val)
            else:
                val = int(self.data[self.ptr])
                self.ptr += 1
                return (3, val)

    parser = Parser(input_data)
    # Create a range of indices to drive the map, though we don't know the exact 
    # number of iterations needed because query lengths vary.
    # Actually, we can just use the total number of elements as an upper bound.
    queries = map(parser.next_query, range(Q))

    def process_step(state, query):
        current_height, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant new flower: birth height is the current total height
            # We use a sorted list to keep track of birth heights
            # Since we can't use .append() in a way that avoids mutation, 
            # but we can mutate the list object.
            import bisect
            bisect.insort(plants, current_height)
            return (current_height, plants, results)
        
        elif q_type == 2:
            # Increase height
            return (current_height + val, plants, results)
        
        else:
            # Harvest plants where current_height - birth_height >= H
            # birth_height <= current_height - H
            import bisect
            threshold = current_height - val
            idx = bisect.bisect_right(plants, threshold)
            harvested_count = idx
            # Remove harvested plants
            del plants[:idx]
            results.append(str(harvested_count))
            return (current_height, plants, results)

    # Initial state: (total_height, sorted_plants_list, results_list)
    initial_state = (0, [], [])
    final_state = reduce(process_step, queries, initial_state)
    
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()