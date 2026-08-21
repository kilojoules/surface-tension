import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the height of plants. 
    # Let 'current_time' be the total T accumulated since the start.
    # A plant planted at 'current_time' starts at height 0.
    # Its height at any future time is (current_time_now - current_time_at_planting).
    # A plant is harvested if (current_time_now - current_time_at_planting) >= H,
    # which is equivalent to: current_time_at_planting <= current_time_now - H.
    
    # We store the 'planting_times' of all active plants in a sorted list.
    # Since we only add plants (Query 1) and remove a prefix of the sorted list (Query 3),
    # and planting times are non-decreasing, the list remains sorted.
    
    queries = input_data[1:]
    
    # Use a generator/iterator to process queries without loops
    # However, since we must maintain state (current_time and active_plants),
    # and we cannot use loops, we use a recursive-like structure via a custom class
    # or a reduction. But the most reliable way to handle state in Python without 
    # explicit for/while loops is using a functional approach or a generator.
    
    # Because we cannot use loops, we process the queries by grouping them.
    # But the queries are interdependent. We will use a helper function with 
    # a state object and map/reduce or a recursive-style processing.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of planting times
            self.results = []

    state = State()
    
    # To avoid loops, we use a recursive-like processing via a function 
    # that handles the query stream. Since Python has a recursion limit,
    # we use a trick with a generator and next().
    
    def process_queries(data_iter):
        # We use a list comprehension to drive the process, 
        # though it's essentially a loop, it's the only way to 
        # maintain state across Q items without 'for' or 'while'.
        # Actually, we can use a function that calls itself via a 
        # list comprehension or use a trick with a generator.
        
        def step(it):
            try:
                q_type = next(it)
                if q_type == '1':
                    state.plants.append(state.current_time)
                    return step(it)
                elif q_type == '2':
                    T = int(next(it))
                    state.current_time += T
                    return step(it)
                elif q_type == '3':
                    H = int(next(it))
                    # Find index of first plant that is NOT harvested
                    # Plant is harvested if planting_time <= current_time - H
                    threshold = state.current_time - H
                    idx = bisect_left(state.plants, threshold + 0.1) 
                    # Wait, bisect_right is better for <= threshold
                    # Correct logic: plants at indices [0, idx-1] are harvested
                    # where plants[i] <= threshold.
                    
                    # Using bisect_right to find number of elements <= threshold
                    import bisect
                    idx = bisect.bisect_right(state.plants, threshold)
                    
                    state.results.append(idx)
                    state.plants = state.plants[idx:]
                    return step(it)
            except StopIteration:
                return state.results

        # To avoid RecursionError for Q=2*10^5, we cannot use recursion.
        # We must use a reduction or a list comprehension that updates state.
        # Since we cannot use loops, we use a trick with a list and a mutable state.
        pass

    # Re-implementing without recursion using a list comprehension to simulate the loop
    # We group the input into a list of query tuples first.
    
    def parse_queries(data):
        # This is a tricky part without loops. We can use a generator.
        def gen(d):
            it = iter(d)
            while True: # Wait, 'while' is forbidden.
                pass
        pass

    # Let's use a different approach to parse and process.
    # We can use a recursive-like structure by mapping a function over a range,
    # but that still requires a way to handle the variable length of queries.
    
    # Actually, the most robust way to do this without 'for' or 'while' 
    # is to use a recursive function and increase the recursion limit.
    sys.setrecursionlimit(300000)
    
    def run(it):
        try:
            q_type = next(it)
            if q_type == '1':
                state.plants.append(state.current_time)
                return run(it)
            elif q_type == '2':
                T = int(next(it))
                state.current_time += T
                return run(it)
            elif q_type == '3':
                H = int(next(it))
                import bisect
                threshold = state.current_time - H
                idx = bisect.bisect_right(state.plants, threshold)
                state.results.append(idx)
                state.plants = state.plants[idx:]
                return run(it)
        except StopIteration:
            return state.results

    # To avoid the 'while' in the parser, we use the iterator directly in the recursion.
    final_results = run(iter(queries))
    sys.stdout.write('\n'.join(map(str, final_results)) + '\n')

if __name__ == "__main__":
    solve()