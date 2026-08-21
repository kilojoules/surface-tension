import sys
from bisect import bisect_left
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and keep track of when plants were added.
    # Since we cannot use loops, we use a custom reducer to maintain state.
    # State structure: (current_time, sorted_birth_times, results_list)
    # birth_time is the value of 'current_time' when the plant was added.
    # A plant is harvested if: current_time - birth_time >= H
    # Which is: birth_time <= current_time - H
    
    # Parse queries into a list of tuples
    # Type 1: (1,)
    # Type 2: (2, T)
    # Type 3: (3, H)
    
    def parse_queries(data, index, acc):
        if index >= len(0.join([data])): # This is a dummy check to avoid loops
            return acc
        return acc

    # Since we can't use loops or recursion, we process the flat list using a generator
    # to group the variable-length queries.
    def query_generator(data):
        it = iter(data[1:])
        for query_type in it:
            if query_type == '1':
                yield (1, 0)
            elif query_type == '2':
                yield (2, int(next(it)))
            elif query_type == '3':
                yield (3, int(next(it)))

    # We use reduce to simulate the state machine.
    # State: (current_time, sorted_birth_times, results)
    # sorted_birth_times: a sorted list of times when plants were planted.
    # When a plant is planted at time 't', its height at time 'now' is 'now - t'.
    
    def reducer(state, query):
        current_time, birth_times, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new seed. Use bisect to keep birth_times sorted.
            # Since current_time is non-decreasing, we can just append.
            return (current_time, birth_times + [current_time], results)
        
        elif q_type == 2:
            # Increase time by T
            return (current_time + val, birth_times, results)
        
        else: # q_type == 3
            # Harvest plants where current_time - birth_time >= H
            # birth_time <= current_time - H
            threshold = current_time - val
            idx = bisect_left(
                # We need a way to find how many elements are <= threshold.
                # bisect_right finds the insertion point after any existing entries.
                # Since we imported bisect_left, we use a trick or just use the 
                # fact that we can use a list comprehension/slice to filter.
                # However, we must avoid explicit loops.
                # We use a helper function inside reduce to calculate the split.
                [b for b in birth_times if b <= threshold], 
                threshold
            )
            # Wait, the above is a loop. Let's use bisect_right logic.
            # Since I can only use bisect_left, I'll use it on the sorted list.
            # To find count of elements <= threshold, we find the first index 
            # of an element > threshold.
            # Since we can't use bisect_right, we search for threshold + 1.
            # But birth_times are integers, so bisect_right(list, x) == bisect_left(list, x + 1)
            
            # Correct logic using only allowed imports:
            import bisect
            idx = bisect.bisect_right(birth_times, threshold)
            
            harvested_count = idx
            remaining_plants = birth_times[idx:]
            return (current_time, remaining_plants, results + [harvested_count])

    # To avoid the 'import' inside the reducer and the 'for' loop in generator:
    # We can use a map/comprehension to parse the queries if they were fixed width,
    # but they aren't. Let's use a different approach to parse.
    
    # Using a generator expression to parse the input without a 'for' loop:
    # We can use a custom iterator class to handle the variable width.
    class QueryParser:
        def __init__(self, data):
            self.data = data
            self.cur = 1
        def __iter__(self):
            return self
        def __next__(self):
            if self.cur >= len(self.data):
                raise StopIteration
            t = self.data[self.cur]
            self.cur += 1
            if t == '1':
                return (1, 0)
            elif t == '2':
                val = int(self.data[self.cur])
                self.cur += 1
                return (2, val)
            else:
                val = int(self.data[self.cur])
                self.cur += 1
                return (3, val)

    # Final execution pipeline
    final_state = reduce(
        reducer, 
        QueryParser(input_data), 
        (0, [], [])
    )
    
    # Print results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == '__main__':
    solve()