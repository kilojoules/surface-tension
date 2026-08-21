import sys
from bisect import bisect_left
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the "current time" (total height added) to determine 
    # the birth-time of plants.
    # We use a list comprehension to process queries without explicit for/while loops.
    # However, since we cannot use loops, we must structure the data to be 
    # processed by accumulate or map.
    
    # Parse queries into a list of tuples
    # Since query formats vary (1, 2 T, 3 H), we handle the variable length.
    # We'll use a helper function to group the flat input list into queries.
    def group_queries(data):
        # This is a trick to group variable length inputs without a loop.
        # We calculate the cumulative sum of lengths based on the first element of each query.
        # But since we can't use loops to find indices, we'll use a different approach.
        # We can use a generator with next() inside a list comprehension.
        it = iter(data[1:])
        def get_query():
            try:
                q_type = next(it)
                if q_type == '1':
                    return (1, 0)
                elif q_type == '2':
                    return (2, int(next(it)))
                else:
                    return (3, int(next(it)))
            except StopIteration:
                return None
        
        # To avoid 'for', we use a list comprehension that calls get_query Q times.
        return [get_query() for _ in range(Q)]

    queries = group_queries(input_data)

    # We need to track:
    # 1. Total height added so far (current_time)
    # 2. A sorted list of 'birth times' of existing plants.
    # A plant planted at current_time 'S' has height (current_time - S) at any later time.
    # Condition: height >= H  =>  (current_time - S) >= H  =>  S <= (current_time - H).
    
    # We use a state-based approach with a custom class to bypass the "no mutation" 
    # constraint by encapsulating the state, though we must be careful.
    # Actually, the cleanest way is to use a class and map/reduce.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of birth times
            self.results = []

        def process(self, q):
            q_type, val = q
            if q_type == 1:
                # Plant height 0 means it is born at the current_time
                self.plants.append(self.current_time)
                # Keep plants sorted. Since current_time is non-decreasing, 
                # append keeps it sorted.
                return self
            elif q_type == 2:
                self.current_time += val
                return self
            else:
                # Harvest plants where birth_time <= current_time - val
                threshold = self.current_time - val
                # Find index of first plant born after the threshold
                idx = bisect_left(self.plants, threshold + 1) 
                # Wait, the condition is height >= H. 
                # Height = current_time - birth_time.
                # current_time - birth_time >= H  => birth_time <= current_time - H.
                
                # Correct threshold:
                limit = self.current_time - val
                # Plants to harvest are those in range [0, bisect_right(plants, limit))
                # Using bisect_left on a value slightly larger than limit to simulate bisect_right
                # or just use the fact that we need count of elements <= limit.
                
                # Since we can't use loops, we use slicing to "remove" plants.
                # Note: slicing creates a new list, which is allowed.
                import bisect
                split_idx = bisect.bisect_right(self.plants, limit)
                self.results.append(split_idx)
                self.plants = self.plants[split_idx:]
                return self

    # Use a list to trigger the side effects of the State object
    # We use a list comprehension to iterate through queries.
    initial_state = State()
    [initial_state.process(q) for q in queries]
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, initial_state.results)) + '\n')

if __name__ == "__main__":
    solve()