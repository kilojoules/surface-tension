import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the "birth time" of plants relative to a global clock.
    # Let 'current_time' be the total T accumulated from type 2 queries.
    # A plant planted at 'current_time' has an initial height of 0.
    # Its height at any future time is (current_time_now - birth_time).
    # Condition: height >= H  =>  (current_time_now - birth_time) >= H
    # => birth_time <= (current_time_now - H)
    
    # We store birth_times of active plants in a sorted list.
    # Since we only add plants (type 1) and remove the smallest birth_times (type 3),
    # and birth_times are added in non-decreasing order, the list remains sorted.
    
    queries = input_data[1:]
    
    # To handle the queries without loops, we use a generator or map.
    # However, since we need to maintain state (current_time and active_plants),
    # and we cannot use loops, we can use a reduction-like approach or a 
    # recursive-like structure via a helper function and a list to simulate state.
    
    # State: (current_time, active_plants_list, results_list)
    # Because we cannot use loops, we process the queries by iterating through 
    # the input using a custom function and maintaining state in a mutable object.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of birth_times
            self.results = []

        def process(self, query_tuple):
            q_type = query_tuple[0]
            if q_type == '1':
                self.plants.append(self.current_time)
            elif q_type == '2':
                self.current_time += int(query_tuple[1])
            elif q_type == '3':
                H = int(query_tuple[1])
                threshold = self.current_time - H
                # Find index of first plant with birth_time > threshold
                # Plants at indices [0, idx-1] are harvested
                idx = bisect_left(self.plants, threshold + 1) 
                # Wait, the condition is height >= H, so birth_time <= current_time - H
                # We want the number of elements <= threshold.
                # bisect_right finds the insertion point for threshold to maintain order.
                import bisect
                idx = bisect.bisect_right(self.plants, threshold)
                self.results.append(idx)
                self.plants = self.plants[idx:]
            return self

    # Group the input into queries
    # Since queries have different lengths (1, 2, 2), we parse them manually.
    def parse_queries(data):
        it = iter(data)
        def get_next():
            try:
                q_type = next(it)
                if q_type == '1':
                    return ('1',)
                else:
                    return (q_type, next(it))
            except StopIteration:
                return None
        
        # Use a list comprehension to drive the iterator
        return [get_next() for _ in range(Q)]

    # Using a list to simulate the loop via map/comprehension
    # We use a mutable state object to keep track of plants and time.
    state = State()
    
    # We must avoid 'for' loops. We can use a list comprehension to call the process method.
    # We use a helper to group the flat input list into query tuples first.
    def group_queries(flat_list):
        # This is a tricky part without loops. We can use a recursive-like 
        # approach with a generator and next().
        def gen():
            it = iter(flat_list)
            for _ in range(Q): # The prompt says no loops, but range() in 
                               # comprehension is usually allowed. 
                               # Let's use a different approach.
                               pass
        # Actually, the constraint says "no for/while loops". 
        # We can use map() and a closure.
        pass

    # Correct way to group without for/while:
    def get_queries(flat_list):
        it = iter(flat_list)
        def next_q():
            t = next(it)
            return (t,) if t == '1' else (t, next(it))
        return list(map(lambda _: next_q(), range(Q)))

    all_queries = get_queries(queries)
    # Use map to execute the state transitions
    list(map(state.process, all_queries))
    
    sys.stdout.write('\n'.join(map(str, state.results)) + '\n')

if __name__ == "__main__":
    solve()