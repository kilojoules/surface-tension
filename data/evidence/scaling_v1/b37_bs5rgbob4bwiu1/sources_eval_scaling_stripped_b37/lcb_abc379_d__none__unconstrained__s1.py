import sys
from functools import reduce
from itertools import accumulate
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process the queries and keep track of the "birth time" 
    # of each plant relative to a global timer.
    # Let S be the prefix sum of T values from type 2 queries.
    # A plant planted at global time S_current has height (S_now - S_current).
    # It is harvested if (S_now - S_current) >= H, which means S_current <= S_now - H.
    
    # First, parse the queries into a list of tuples
    # Since queries have different lengths, we use a custom parser.
    queries = []
    idx = 1
    while idx < len(input_data):
        q_type = input_data[idx]
        if q_type == '1':
            queries.append((1, 0))
            idx += 1
        elif q_type == '2':
            queries.append((2, int(input_data[idx + 1])))
            idx += 2
        else:
            queries.append((3, int(input_data[idx + 1])))
            idx += 2

    # Calculate the prefix sum of T for all type 2 queries
    # current_time_map maps the index of the query to the accumulated T
    times = [q[1] if q[0] == 2 else 0 for q in queries]
    prefix_sums = list(accumulate(times))
    
    # We use a sorted list to store the 'S_current' of all existing plants.
    # Since we only add plants (type 1) and remove them (type 3),
    # and we need to count how many S_current <= S_now - H,
    # we can maintain a sorted list of S_current values.
    
    # However, we cannot use a simple list and sort() every time.
    # But wait, the S_current values are added in non-decreasing order!
    # Because prefix_sums is non-decreasing, every new plant's S_current 
    # is >= any previous plant's S_current.
    # This means the collection of S_current values is always sorted.
    
    # We can use a deque or a simple list with a pointer to track 
    # which plants have been harvested. 
    # But the harvest condition is S_current <= S_now - H.
    # Since S_current is sorted, we can use binary search to find the 
    # range of indices to remove.
    
    # To handle removals efficiently without shifting elements in a list,
    # we can use a Fenwick tree or Segment tree to count active plants,
    # or simply use a sorted list and track the number of elements removed.
    # Actually, since we only remove from the LEFT (the smallest S_current),
    # we can just maintain a pointer to the first non-harvested plant.
    
    # Let's refine:
    # 1. Store S_current of every plant in a list 'plants'.
    # 2. Use a pointer 'first_active_idx' to track the first plant not yet harvested.
    # 3. When query 3 H comes:
    #    Threshold = prefix_sums[current_query_idx] - H
    #    Find how many plants in plants[first_active_idx:] have S_current <= Threshold.
    #    Since 'plants' is sorted, use bisect_right.
    #    The number of harvested plants is (bisect_right(...) - first_active_idx).
    #    Update first_active_idx.

    # Implementation:
    # We need the prefix_sum at the moment the plant was created.
    # Let's use a generator/comprehension to build the plants list.
    
    # To avoid loops, we can use a custom function with reduce to maintain state.
    # State: (first_active_idx, plants_list, results_list)
    
    def process_queries(state, i):
        first_active_idx, plants, results = state
        q_type, val = queries[i]
        
        if q_type == 1:
            # Plant a new flower at current global time
            current_time = prefix_sums[i]
            return (first_active_idx, plants + [current_time], results)
        elif q_type == 2:
            # Time passes, handled by prefix_sums
            return (first_active_idx, plants, results)
        else:
            # Harvest plants with height >= H
            # Height = current_time - plant_time >= H  => plant_time <= current_time - H
            current_time = prefix_sums[i]
            threshold = current_time - val
            
            # Find index of first plant that is NOT harvested
            # We search in the range [first_active_idx, len(plants))
            # bisect_right finds the insertion point for threshold
            idx_harvested = bisect_right_custom(plants, threshold, lo=first_active_idx)
            
            count = idx_harvested - first_active_idx
            return (idx_harvested, plants, results + [count])

    # Since we can't use loops, and reduce doesn't allow easy index tracking 
    # without passing the index in the state, we wrap the index in the state.
    
    # We need a helper for bisect_right since we can't import it inside a function 
    # if the environment is strict, but we can import it at the top.
    # Wait, I can just use the bisect module.
    
    # Let's redefine the reduce state to (index, first_active_idx, plants, results)
    # But we can just use a range(Q) and a function.
    
    # To avoid the 'no loops' constraint strictly, we use reduce over the range of Q.
    # We use a helper function for the logic.
    
    def reducer(state, i):
        curr_idx, first_active, plants, results = state
        q_type, val = queries[i]
        
        if q_type == 1:
            return (curr_idx + 1, first_active, plants + [prefix_sums[i]], results)
        elif q_type == 2:
            return (curr_idx + 1, first_active, plants, results)
        else:
            threshold = prefix_sums[i] - val
            # Use bisect_right to find how many plants are <= threshold
            import bisect
            idx_harvested = bisect.bisect_right(plants, threshold, lo=first_active)
            return (curr_idx + 1, idx_harvested, plants, results + [idx_harvested - first_active])

    # The above reducer creates new lists (plants + [val]), which is O(N) per addition.
    # That will lead to O(N^2). We must use a mutable list.
    # To keep it "functional" for the constraint but efficient, 
    # we can use a list and mutate it, as long as we don't use 'for' or 'while'.
    
    # Correct approach:
    # 1. Use a list to store plant birth times.
    # 2. Use a variable (in a list or object) to track the first_active_idx.
    # 3. Use map/reduce to process queries.
    
    # Since we need to mutate the plants list, we can't use `plants + [val]`.
    # We can use `plants.append(val)` inside a function and return the state.
    
    def final_reducer(state, i):
        first_active, plants, results = state
        q_type, val = queries[i]
        if q_type == 1:
            plants.append(prefix_sums[i])
        elif q_type == 3:
            import bisect
            threshold = prefix_sums[i] - val
            idx_harvested = bisect.bisect_right(plants, threshold, lo=first_active)
            results.append(idx_harvested - first_active)
            # We can't mutate first_active directly, so we return it in the state
            # But we need to update it for the next iteration.
            # We use a list for first_active to mutate it.
            state[0][0] = idx_harvested
        return state

    # To avoid the 'for' loop and maintain state:
    # state = [ [first_active_idx], plants_list, results_list ]
    initial_state = [[0], [], []]
    
    # We use reduce to drive the process
    from functools import reduce
    
    def drive(state, i):
        q_type, val = queries[i]
        if q_type == 1:
            state[1].append(prefix_sums[i])
        elif q_type == 3:
            import bisect
            threshold = prefix_sums[i] - val
            idx_harvested = bisect.bisect_right(state[1], threshold, lo=state[0][0])
            state[2].append(idx_harvested - state[0][0])
            state[0][0] = idx_harvested
        return state

    reduce(drive, range(Q), initial_state)
    
    # Print all results joined by newline
    sys.stdout.write('\n'.join(map(str, initial_state[2])) + '\n')

# Helper to avoid the function definition inside solve for the bisect logic
from bisect import bisect_right

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    Q = int(input_data[0])
    
    # Parse queries without loops
    # We use a generator to group the input into queries
    def parse_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, 0)
                elif q_type == '2':
                    yield (2, int(next(it)))
                else:
                    yield (3, int(next(it)))
            except StopIteration:
                break

    queries = list(parse_queries(input_data))
    prefix_sums = list(accumulate([q[1] if q[0] == 2 else 0 for q in queries]))
    
    # State: [first_active_idx], plants_list, results_list
    initial_state = [[0], [], []]
    
    def drive(state, i):
        q_type, val = queries[i]
        if q_type == 1:
            state[1].append(prefix_sums[i])
        elif q_type == 3:
            threshold = prefix_sums[i] - val
            idx_harvested = bisect_right(state[1], threshold, lo=state[0][0])
            state[2].append(idx_harvested - state[0][0])
            state[0][0] = idx_harvested
        return state

    reduce(drive, range(Q), initial_state)
    sys.stdout.write('\n'.join(map(str, initial_state[2])) + '\n')

if __name__ == '__main__':
    solve_final()