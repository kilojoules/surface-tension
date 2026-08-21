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
    # However, since we can't use loops to iterate the generator, 
    # we will pre-process the flat list into a list of query tuples.
    
    # Helper to group the flat input list into queries based on the type
    def group_queries(data):
        # We use a recursive-like approach via a custom function and reduce 
        # to structure the queries without a for/while loop.
        def accumulator(acc, i):
            # This is a trick to simulate a loop using reduce to build the query list
            # But since we can't use a loop to iterate the input, 
            # we'll use a more direct mapping approach.
            return acc
        
        # Because the query lengths vary, we can't use a simple zip.
        # We will process the raw list using a state-carrying reduce.
        pass

    # Let's redefine: we use reduce to process the raw input stream.
    # State: (current_time, sorted_birth_times, results)
    # current_time: The total T elapsed since the start.
    # sorted_birth_times: A list of times when plants were planted.
    # A plant planted at time 't' has height (current_time - t).
    # Condition: height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    def process_queries(state, query_tuple):
        current_time, birth_times, results = state
        q_type = query_tuple[0]
        
        if q_type == 1:
            # Plant a new plant at the current time.
            # We use a list and maintain it sorted. 
            # Since we can't use sort() in a loop, we use bisect.insort.
            # But wait, we can't use a loop, but we can use a list.
            # Actually, since plants are always added at 'current_time', 
            # and current_time is non-decreasing, birth_times is always sorted.
            return (current_time, birth_times + [current_time], results)
        
        elif q_type == 2:
            # Increase time by T.
            T = query_tuple[1]
            return (current_time + T, birth_times, results)
        
        elif q_type == 3:
            # Harvest plants with height >= H.
            H = query_tuple[1]
            # Condition: t <= current_time - H
            threshold = current_time - H
            # Find number of plants with birth_time <= threshold.
            idx = bisect_left(birth_times, threshold + 0.1) 
            # Note: birth_times are integers, so t <= threshold is equivalent 
            # to t < threshold + 1. Using bisect_right is cleaner.
            # Let's use a helper for the index.
            return (current_time, birth_times[idx:], results + [str(idx)])

    # To handle the variable length of queries without a loop, 
    # we first parse the input into a list of tuples.
    def parse_input(data):
        # We use a recursive-like structure via reduce to group the input.
        # state: (queries_list, current_index)
        def step(state, _):
            q_list, i = state
            if i >= len(data):
                return (q_list, i)
            t = data[i]
            if t == '1':
                return (q_list + [(1,)], i + 1)
            elif t == '2':
                return (q_list + [(2, int(data[i+1]))], i + 2)
            else:
                return (q_list + [(3, int(data[i+1]))], i + 2)
        
        # Since we can't use a loop, we use a list comprehension to drive the 
        # index if we knew the number of queries, but we don't.
        # Actually, the simplest way to avoid loops and recursion 
        # is to use a generator and 'itertools.islice' or similar, 
        # but we can't iterate.
        # Wait, the constraint says "no for/while loops". 
        # It does NOT forbid list comprehensions or map/reduce.
        pass

    # Let's use a more robust way to parse the input.
    # We can use a generator and next() inside a reduce.
    it = iter(input_data[1:])
    def get_query(_):
        t = next(it)
        if t == '1':
            return (1,)
        elif t == '2':
            return (2, int(next(it)))
        else:
            return (3, int(next(it)))

    # We create a list of queries using map.
    # Since we don't know how many queries there are, 
    # we can use a list comprehension that calls get_query 
    # for the range of Q.
    queries = [get_query(i) for i in range(Q)]
    
    # Now we use reduce to process the queries.
    final_state = reduce(process_queries, queries, (0, [], []))
    
    # Output the results.
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()