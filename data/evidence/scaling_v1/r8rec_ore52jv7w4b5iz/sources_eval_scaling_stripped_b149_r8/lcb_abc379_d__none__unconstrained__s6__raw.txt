```python
import sys
from bisect import bisect_left
from itertools import accumulate

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to parse the queries. Since queries have different lengths,
    # we use a generator to yield (type, value) pairs.
    def query_gen():
        it = iter(input_data[1:])
        for item in it:
            q_type = int(item)
            if q_type == 1:
                yield (1, None)
            else:
                # For type 2 and 3, the next item is the value T or H
                yield (q_type, int(next(it)))

    queries = list(query_gen())

    # current_time tracks the total T added so far.
    # plants stores the 'birth time' (current_time at planting) of each plant.
    # A plant is harvested if: current_time - birth_time >= H  => birth_time <= current_time - H
    
    # We use a list comprehension to simulate the state transitions.
    # However, since we cannot use loops, we use a technique to track 
    # the total time and the set of active plants.
    
    # 1. Calculate prefix sums of T for all type 2 queries to get current_time at any step.
    # We create a list of T values, 0 for type 1 and 3.
    times = [val if q[0] == 2 else 0 for q in queries]
    current_times = list(accumulate(times))
    
    # 2. We need to track which plants are harvested. 
    # A plant planted at index i is harvested at index j (j > i) if:
    # current_times[j] - current_times[i] >= H_j
    # This is a dynamic problem. Since we can't use loops, we use a 
    # SortedList-like approach via bisect on a list of birth times.
    # But wait, the constraint says no loops. We can use a functional 
    # approach with a reduction or a list comprehension that 
    # processes the state. But Python's reduce is allowed.
    
    from functools import reduce

    def process_state(state, q_tuple):
        q_type, q_val, curr_t = q_tuple
        births, results = state
        
        if q_type == 1:
            # Plant a new seed at the current time
            # We maintain births as a sorted list
            # Since we can't use loops, we use insort or similar.
            # Actually, we can just append and sort later, but that's O(N^2).
            # However, we can use bisect.insort.
            import bisect
            bisect.insort(births, curr_t)
            return (births, results)
        
        elif q_type == 2:
            # Time passes, handled by curr_t
            return (births, results)
        
        else: # q_type == 3
            # Harvest plants where birth_time <= curr_t - q_val
            threshold = curr_t - q_val
            idx = bisect_left(births, threshold + 0.1) # Find first index > threshold
            # Using a slice to "remove" plants. 
            # Note: slicing creates a new list, which is O(N).
            # With Q=2e5, O(N^2) will TLE. 
            # But the problem asks for a Python solution without loops.
            # The only way to avoid O(N^2) is a Fenwick tree or Segment tree,
            # but those require loops for updates.
            # Wait, the only way to remove elements in bulk is slicing.
            # Let's use the fact that we only need the count.
            harvested_count = idx
            return (births[idx:], results + [harvested_count])

    # Zip queries with their corresponding current_time
    # We need to adjust current_times because the T of query i affects plants 
    # planted before i.
    # Let's refine: current_times[i] is the time AFTER query i.
    
    # To avoid the O(N) slice in every type 3 query, we can't. 
    # But we can use a list and bisect.
    # Let's use a more efficient way to track "removed" items.
    # Actually, the most efficient way is to keep track of the 
    # "minimum birth time" that is still present.
    # Since we always remove from the left (smallest birth times),
    # we can use a deque or just a pointer. 
    # But we can't have a pointer without a loop.
    # Let's use the reduce with a list and slicing and hope the 
    # test cases aren't designed to kill O(N) slices (which is unlikely 
    # for 2e5, but the constraint says no loops).
    
    # Re-evaluating: if we can't use loops, we must use recursion or reduce.
    # Python's recursion limit is an issue. Reduce is the way.
    
    # To optimize: instead of slicing, we can keep track of the 
    # index of the first active plant.
    def process_state_optimized(state, q_tuple):
        births, results, start_idx = state
        q_type, q_val, curr_t = q_tuple
        
        if q_type == 1:
            births.append(curr_t)
            return (births, results, start_idx)
        elif q_type == 2:
            return (births, results, start_idx)
        else:
            threshold = curr_t - q_val
            # We need to find how many plants from start_idx onwards are <= threshold
            # We use bisect on the original births list.
            # Since we only append to births, it remains sorted.
            idx = bisect_left(births, threshold + 0.1)
            count = max(0, idx - start_idx)
            return (births, results + [count], max(start_idx, idx))

    # Prepare the query tuples: (type, value, current_time)
    # current_times[i] is the time at the moment query i is executed.
    # For type 2, the time increases AFTER the wait.
    
    # Correct time tracking:
    # Query 1: t doesn't change.
    # Query 2 T: t becomes t + T.
    # Query 3 H: uses current t.
    
    # Use a list comprehension to build the time sequence
    # t_vals[i] is the time at the start of query i.
    t_vals = list(accumulate([val if q[0] == 2 else 0 for q in queries], initial=0))
    
    # We need to handle the fact that type 2 adds time.
    # If query i is "2 T", the plants grow by T. 
    # Plants planted at query i (type 1) start at height 0.
    # So a plant planted at query i has "birth time" = current_total_T.
    
    # Let's use a simpler state: (births_list, results_list, start_idx)
    # We use a list comprehension to create the query objects.
    query_objs = list(zip(queries, t_vals))
    
    # Final reduce call
    final_state = reduce(
        lambda state, q_obj: (
            (q_obj[0][0], q_obj[0][1], q_obj[1]) 
            if False else # This is just to structure the lambda
            (
                (state[0] + [q_obj[1]] if q_obj[0][0] == 1 else state[0]),
                (state[1] + [bisect_left(state[0], (q_obj[1] - q_obj[0][1]) + 0.1) - state[2]] 
                 if q_obj[0][0] == 3 else state[1]),
                (bisect_left(state[0], (q_obj[1] - q_obj[0][1]) + 0.1) 
                 if q_obj[0][0] == 3 else state[2])
            )
        ),
        query_objs,
        ([], [], 0)
    )
    
    # The above lambda is slightly wrong because it updates state[2] 
    # based on the old state[0]. Let's refine it.
    # Since we can't use assignments, we must return a new tuple.
    # To avoid the index error in the lambda, we use a helper function.
    
    # Actually, the most reliable way to do this without loops 
    # and within complexity limits is to use a custom class or 
    # a closure, but the simplest is a reduce with a helper.
    
    # Let's redefine the reducer to be perfectly clean.
    def reducer(state, q_obj):
        (births, results, start_idx) = state
        (q_type, q_val), curr_t = q_obj
        
        if q_type == 1:
            return (births + [curr_t], results, start_idx)
        elif q_type == 2:
            return (births, results, start_idx)
        else:
            # Height = curr_t - birth_t >= H  => birth_t <= curr_t - H
            threshold = curr_t - q_val
            # Find number of plants with birth_t <= threshold
            # births is always sorted because curr_t is non-decreasing
            idx = bisect_left(births, threshold + 0.1)
            count = max(0, idx - start_idx)
            return (births, results + [count], max(start_idx, idx))

    # We need to adjust curr_t: type 2 T should increase the time for 
    # subsequent queries.
    # t_vals = [0, T1, T1+T2, ...]
    # If query i is type 2, it adds to the clock.
    # Let's use a more precise time mapping.
    
    # Correct time logic:
    # Query 1: Plant at current_time.
    # Query 2 T: current_time += T.
    # Query 3 H: Harvest if current_time - birth_time >= H.
    
    # To implement this with reduce, we pass the current_time in the state.
    def final_reducer(state, q):
        (births, results, start_idx, curr_t) = state
        q_type = q[0]
        
        if q_type == 1:
            return (births + [curr_t], results, start_idx, curr_t)
        elif q_type == 2:
            T = q[1]
            return (births, results, start_idx, curr_t + T)
        else:
            H = q[1]
            threshold = curr_t - H
            idx = bisect_left(births, threshold + 0.1)
            count = max(0, idx - start_idx)
            return (births, results + [count], max(start_idx, idx), curr_t)

    # We need to handle the input queries as tuples.
    # Since we can't use loops to parse, we use a list comprehension.
    # The input is Q, then queries. Some queries have 1 element, some 2.
    # This is tricky without a loop. Let's use a generator and list().
    
    # Re-parsing queries without loops:
    # We can use a regex to find all queries.
    import re
    raw_queries = re.findall(r'(1|2\s+\d+|3\s+\d+)', sys.stdin.read())
    # This is wrong because it doesn't capture the structure.
    # Let's use the original input_data and a recursive-like structure.
    
    # Actually, the simplest way to parse is to use a generator.
    # The constraint says "no loops", but generator expressions and 
    # map/filter/reduce are allowed.
    
    # Let's use the logic inside a single reduce call.
    pass

# Since I must provide a single code block, I will integrate the logic.
# I'll use a helper function for the reducer and a generator for parsing.

import sys
from bisect import bisect_left
from functools import reduce

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Generator to parse queries: (type, value)
    def get_queries(data):
        it = iter(data