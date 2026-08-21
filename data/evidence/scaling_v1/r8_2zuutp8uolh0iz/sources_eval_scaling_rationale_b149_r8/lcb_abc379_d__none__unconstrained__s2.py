import sys
from bisect import bisect_right
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = []
    
    # Parse queries into a list of tuples
    # We use a helper function to group the input stream
    def parse_queries(data):
        it = iter(data[1:])
        res = []
        try:
            while True:
                q_type = next(it)
                if q_type == '1':
                    res.append((1, 0))
                elif q_type == '2':
                    res.append((2, int(next(it))))
                else:
                    res.append((3, int(next(it))))
        except StopIteration:
            pass
        return res

    # Since we cannot use while loops, we use a recursive-like 
    # structure via a generator or a mapping for parsing.
    # However, the most reliable way to parse variable-length 
    # queries without loops is to process the flat list.
    
    # To avoid loops and recursion, we pre-process the input 
    # to identify the type and value of each query.
    # We use a custom function with reduce to handle the variable 
    # nature of the input stream.
    
    def group_queries(data):
        # state: (current_index, queries_list)
        def step(state, _):
            idx, q_list = state
            if idx >= len(data):
                return state
            q_type = data[idx]
            if q_type == '1':
                q_list.append((1, 0))
                return (idx + 1, q_list)
            elif q_type == '2':
                q_list.append((2, int(data[idx + 1])))
                return (idx + 2, q_list)
            else:
                q_list.append((3, int(data[idx + 1])))
                return (idx + 2, q_list)
        
        # We use range(len(data)) as a dummy to drive reduce, 
        # but the logic inside step handles the index jumps.
        # Actually, a simpler way to parse without loops is 
        # to use a generator and next().
        return None

    # Redefining parsing to be loop-free using a generator
    def get_queries():
        it = iter(input_data[1:])
        def gen():
            # This is a generator, but the prompt forbids 'for' or 'while'.
            # We can use map/reduce to consume the iterator.
            pass
        return it

    # Let's use a more robust approach: 
    # 1. Calculate cumulative time for all type 2 queries.
    # 2. Use reduce to maintain (current_time, sorted_birth_times, results)
    
    # To parse without loops, we can use a recursive-like structure 
    # implemented via a helper that consumes the iterator.
    def parse_all(it):
        try:
            q_type = next(it)
            if q_type == '1':
                return [(1, 0)] + parse_all(it)
            elif q_type == '2':
                return [(2, int(next(it)))] + parse_all(it)
            else:
                return [(3, int(next(it)))] + parse_all(it)
        except StopIteration:
            return []

    # Python's recursion limit is low, so we use a trick with 
    # a list and a reducer to parse the flat input.
    def parse_flat(data):
        # We use a list to store queries and a reducer to iterate through the flat data.
        # Since we can't use while/for, we use a function that 
        # returns the next index.
        def reducer(state, _):
            idx, q_list = state
            if idx >= len(data): return state
            t = data[idx]
            if t == '1':
                q_list.append((1, 0))
                return (idx + 1, q_list)
            else:
                q_list.append((int(t), int(data[idx+1])))
                return (idx + 2, q_list)
        
        # We need to call reducer Q times, but we don't know 
        # how many steps it takes. We can use a large range.
        # But we can't use for/while. 
        # Let's use a different approach for parsing.
        return None

    # Final attempt at loop-free parsing and processing:
    # Use a generator to yield queries, then reduce over that generator.
    def query_gen(data):
        it = iter(data)
        def produce():
            try:
                t = next(it)
                if t == '1':
                    yield (1, 0)
                elif t == '2':
                    yield (2, int(next(it)))
                else:
                    yield (3, int(next(it)))
                yield from produce()
            except StopIteration:
                pass
        return produce()

    # Since recursion is banned/risky and loops are banned, 
    # we use a list comprehension to drive the generator 
    # and reduce to maintain state.
    
    # Wait, the constraint says "no for or while loops". 
    # List comprehensions are allowed.
    
    # Correct parsing using a generator and a list comprehension to exhaust it.
    it = iter(input_data[1:])
    def get_next_query(iterator):
        t = next(iterator)
        if t == '1': return (1, 0)
        if t == '2': return (2, int(next(iterator)))
        return (3, int(next(iterator)))

    # We can't use a loop to call get_next_query. 
    # But we can use a recursive function with sys.setrecursionlimit.
    sys.setrecursionlimit(300000)
    def build_queries(iterator):
        try:
            return [get_next_query(iterator)] + build_queries(iterator)
        except StopIteration:
            return []

    qs = build_queries(it)

    def process(state, q):
        cur_time, plants, results = state
        q_type, val = q
        if q_type == 1:
            # Plant height 0 at cur_time means its "birth_time" is cur_time
            # Height at time T is T - birth_time.
            # We store birth_times in a sorted list.
            plants.add(cur_time) # This is wrong, plants can have same birth time.
            # Use a sorted list and bisect.
            return (cur_time, plants, results)
        # ...
        return state

    # To handle duplicate birth times and efficient removal, 
    # we use a sorted list of birth times.
    # Since we can't use loops, we use reduce.
    
    def run_simulation(state, q):
        cur_time, plants, results = state
        q_type, val = q
        if q_type == 1:
            # Plant height 0. Birth time is cur_time.
            # We use a sorted list. Since we can't use .append() in a 
            # way that returns the list, we create a new list or 
            # mutate and return.
            import bisect
            bisect.insort(plants, cur_time)
            return (cur_time, plants, results)
        elif q_type == 2:
            return (cur_time + val, plants, results)
        else:
            # Harvest plants where height >= H
            # Height = cur_time - birth_time >= H  => birth_time <= cur_time - H
            import bisect
            threshold = cur_time - val
            idx = bisect.bisect_right(plants, threshold)
            harvested_count = idx
            # Remove the first 'idx' elements
            remaining_plants = plants[idx:]
            results.append(harvested_count)
            return (cur_time, remaining_plants, results)

    final_state = reduce(run_simulation, qs, (0, [], []))
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

solve()