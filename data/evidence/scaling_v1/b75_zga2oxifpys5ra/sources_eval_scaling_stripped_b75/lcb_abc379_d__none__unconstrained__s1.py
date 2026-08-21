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
    # A plant planted at 'current_time' has an initial height of 0.
    # Its height at any future time is (current_time_now - current_time_at_planting).
    # The condition height >= H becomes:
    # current_time_now - current_time_at_planting >= H
    # current_time_at_planting <= current_time_now - H
    
    # We store the 'current_time_at_planting' of all existing plants in a sorted list.
    # Since we only add plants (which always have the current_time), 
    # the list of planting times will naturally be sorted.
    
    queries = input_data[1:]
    
    # Using a generator-like approach to parse the queries without loops
    # However, since we must process queries in order and maintain state,
    # we use a helper function with a closure or a class to simulate the state.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of planting times
            
        def process(self, q_type, val=None):
            if q_type == '1':
                self.plants.append(self.current_time)
                return None
            elif q_type == '2':
                self.current_time += int(val)
                return None
            elif q_type == '3':
                H = int(val)
                # Find plants where planting_time <= current_time - H
                threshold = self.current_time - H
                # bisect_right finds the number of elements <= threshold
                # Since we need to remove them, we find the index.
                idx = bisect_left(self.plants, threshold + 1)
                harvested_count = idx
                # Remove the harvested plants from the list
                self.plants = self.plants[idx:]
                return harvested_count

    state = State()
    
    # Parse the flat list of queries into (type, value) pairs
    # Because queries have different lengths, we use a custom iterator.
    def get_queries(data):
        it = iter(data)
        for item in it:
            if item == '1':
                yield ('1', None)
            elif item == '2':
                yield ('2', next(it))
            elif item == '3':
                yield ('3', next(it))

    # Map the process function over the query generator and filter out None
    results = [state.process(t, v) for t, v in get_queries(queries)]
    
    # Print all non-None results separated by newlines
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()