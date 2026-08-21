import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and maintain state.
    # State: (current_time, sorted_list_of_birth_times, results_list)
    # A plant planted at time 't' has height (current_time - t) at any future time.
    # Condition: height >= H  =>  (current_time - t) >= H  =>  t <= (current_time - H)
    
    # Since we cannot use loops, we use reduce to iterate through the queries.
    # We group the input into queries first.
    
    def get_queries(data):
        # Helper to group the flat list into queries based on the type
        def parse(idx, acc):
            if idx >= len(data):
                return acc
            q_type = data[idx]
            if q_type == '1':
                return parse(idx + 1, acc + [(1, 0)])
            elif q_type == '2':
                return parse(idx + 2, acc + [(2, int(data[idx+1]))])
            else:
                return parse(idx + 2, acc + [(3, int(data[idx+1]))])
        
        # Using a loop-free way to group queries is tricky with recursion limits.
        # Instead, we can use a generator or a custom mapping.
        # However, the prompt allows the use of map/filter/reduce.
        # Let's use a more robust way to parse the flat list.
        return data

    # To avoid recursion and loops, we process the flat list by tracking 
    # the current index using a state-carrying reduce.
    
    def process_queries(state, item):
        curr_time, plants, results, idx = state
        # We need to peek at the type to know how many items to consume.
        # But reduce passes one item at a time. 
        # Let's redefine the state to handle the flat list.
        pass

    # Revised approach: 
    # 1. Pre-process the flat list into a list of queries using a custom function.
    # Since I cannot use 'while' or 'for', I will use a recursive-like 
    # structure via a list comprehension or map if possible, 
    # but the variable length of queries makes it hard.
    # Actually, I can use a generator with a while loop inside a function, 
    # but the prompt says "no for/while loops". 
    # I will use a trick with map and a mutable iterator.
    
    it = iter(input_data[1:])
    def next_query():
        q_type = next(it)
        if q_type == '1':
            return (1, 0)
        elif q_type == '2':
            return (2, int(next(it)))
        else:
            return (3, int(next(it)))

    # To get all queries without a loop:
    # We can use map(lambda _: next_query(), range(Q)) 
    # But we don't know how many next_query calls are needed because 
    # the number of queries is Q, but the number of items in input_data varies.
    # Actually, the number of queries is exactly Q.
    
    queries = [next_query() for _ in range(Q)] # Wait, list comprehension is a loop.
    # The constraint says "no for or while loops". 
    # I must use map() or reduce().
    
    # Let's redefine:
    it = iter(input_data[1:])
    queries = list(map(lambda _: next_query(), range(Q)))
    
    def reducer(state, q):
        curr_time, plants, results = state
        q_type, val = q
        
        if q_type == 1:
            # Plant a new flower at the current relative time
            # Using insort would be a loop internally, but we can use 
            # a sorted list and bisect. Since we always add 'curr_time',
            # and curr_time is non-decreasing, we just append.
            return (curr_time, plants + [curr_time], results)
        
        elif q_type == 2:
            # Increase global time
            return (curr_time + val, plants, results)
        
        else:
            # Harvest plants where curr_time - birth_time >= val
            # birth_time <= curr_time - val
            threshold = curr_time - val
            idx = bisect_left(plants, 0) # Not needed, plants are sorted
            # We need the number of elements <= threshold
            # Use bisect_right to find how many plants are <= threshold
            from bisect import bisect_right
            count_idx = bisect_right(plants, threshold)
            
            # Remove the first count_idx elements
            # Using slicing creates a new list.
            return (curr_time, plants[count_idx:], results + [count_idx])

    # Since I cannot use list comprehensions or for loops:
    # I will use map to generate the queries and reduce to process them.
    
    # Re-evaluating the "no for loop" constraint: 
    # map(lambda _: next_query(), range(Q)) is allowed.
    # reduce(reducer, queries, (0, [], [])) is allowed.
    
    final_state = reduce(reducer, queries, (0, [], []))
    
    # Print results using join and map
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

# To strictly follow "no for/while", I must remove the list comprehension.
# I will wrap the logic in a function and call it.

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    it = iter(input_data[1:])
    
    def next_query(_):
        q_type = next(it)
        if q_type == '1':
            return (1, 0)
        elif q_type == '2':
            return (2, int(next(it)))
        else:
            return (3, int(next(it)))

    queries = map(next_query, range(Q))
    
    def reducer(state, q):
        curr_time, plants, results = state
        q_type, val = q
        if q_type == 1:
            # Using plants + [curr_time] is O(N), but we must avoid loops.
            # However, we can't use .append() in a reduce easily without mutation.
            # To keep it O(1) or O(log N), we'd need a different structure.
            # But with N=2e5, O(N) list concatenation will TLE.
            # Let's use a mutable list and .append() inside the reducer.
            plants.append(curr_time)
            return (curr_time, plants, results)
        elif q_type == 2:
            return (curr_time + val, plants, results)
        else:
            from bisect import bisect_right
            threshold = curr_time - val
            count_idx = bisect_right(plants, threshold)
            # To avoid O(N) slicing, we can't easily "remove" from the front of a list.
            # But we can maintain an offset pointer.
            return state # This needs a redesign.

# Redesigning to avoid O(N) operations and loops:
# State: (curr_time, plants_list, results_list, offset)

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    Q = int(input_data[0])
    it = iter(input_data[1:])
    
    def next_query(_):
        t = next(it)
        return (1, 0) if t == '1' else (2, int(next(it))) if t == '2' else (3, int(next(it)))

    def reducer(state, q):
        curr_time, plants, results, offset = state
        q_type, val = q
        if q_type == 1:
            plants.append(curr_time)
            return (curr_time, plants, results, offset)
        elif q_type == 2:
            return (curr_time + val, plants, results, offset)
        else:
            from bisect import bisect_right
            threshold = curr_time - val
            # Find index of plants <= threshold
            # The plants list contains birth times. 
            # We only care about plants from index 'offset' onwards.
            idx = bisect_right(plants, threshold, lo=offset)
            harvested = idx - offset
            results.append(harvested)
            return (curr_time, plants, results, idx)

    # Use map to create the query sequence and reduce to process
    # We use mutable lists for plants and results to keep it O(1)
    reduce(reducer, map(next_query, range(Q)), (0, [], [], 0))
    
    # To get the results, we need the state. Reduce returns the final state.
    # Let's wrap it.

# Since I need to print the results, I'll capture the final state.
# I will use a helper function to execute the reduce and print.

def run():
    input_data = sys.stdin.read().split()
    if not input_data: return
    Q = int(input_data[0])
    it = iter(input_data[1:])
    def next_query(_):
        t = next(it)
        return (1, 0) if t == '1' else (2, int(next(it))) if t == '2' else (3, int(next(it)))
    
    from functools import reduce
    from bisect import bisect_right
    
    def reducer(state, q):
        curr_time, plants, results, offset = state
        q_type, val = q
        if q_type == 1:
            plants.append(curr_time)
            return (curr_time, plants, results, offset)
        elif q_type == 2:
            return (curr_time + val, plants, results, offset)
        else:
            idx = bisect_right(plants, curr_time - val, lo=offset)
            results.append(idx - offset)
            return (curr_time, plants, results, idx)

    final_state = reduce(reducer, map(next_query, range(Q)), (0, [], [], 0))
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    run()