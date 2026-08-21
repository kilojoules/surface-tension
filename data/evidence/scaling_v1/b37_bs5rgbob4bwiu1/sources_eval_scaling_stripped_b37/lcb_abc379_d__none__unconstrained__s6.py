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
    
    # We need to parse the queries. Since they have different lengths,
    # we process the input stream manually.
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
        elif q_type == '3':
            queries.append((3, int(input_data[idx + 1])))
            idx += 2

    # To handle height increases efficiently, we track the total time passed.
    # A plant planted at total_time 's' has height (current_total_time - s).
    # Condition: height >= H  =>  current_total_time - s >= H  =>  s <= current_total_time - H.
    
    # We maintain a sorted list of 's' values (planting times) for all existing plants.
    # Since we only add plants (s increases monotonically) and remove them from the 
    # left (smallest s), we can use a sorted list and binary search.
    
    # However, we cannot use a simple list.pop(0) as it is O(N).
    # Instead, we keep track of the index of the first non-harvested plant.
    
    # 1. Calculate prefix sums of T for all type 2 queries to get total_time at any point.
    # But T only applies to type 2. Let's map every query to the total time elapsed.
    
    # We can use a generator/map to create a sequence of (type, value)
    # and a running total for time.
    
    def process_queries(qs):
        # current_time tracks the sum of all T from type 2 queries
        # plants stores the 's' (time of planting) for each plant
        # We use a list and a pointer 'start_idx' to simulate a queue
        
        # To avoid loops, we use a custom reducer or a comprehension.
        # Since we need to maintain state (current_time, plants_list, start_idx),
        # and we need to output for every type 3, we can use a generator.
        
        state = {'time': 0, 'plants': [], 'start_idx': 0}
        
        def handle(q):
            t, v = q
            if t == 1:
                state['plants'].append(state['time'])
            elif t == 2:
                state['time'] += v
            elif t == 3:
                # Find how many plants have s <= state['time'] - v
                threshold = state['time'] - v
                # Binary search for the number of plants to harvest
                # We search in the slice plants[start_idx:]
                # bisect_left returns the index in the original list
                idx_harvested = bisect_left(state['plants'], threshold, lo=state['start_idx'])
                
                # The number of harvested plants is (idx_harvested - start_idx)
                # Wait, bisect_left finds the first element >= threshold.
                # We need elements <= threshold. 
                # Actually, we need the index of the first element > threshold.
                # That is bisect_right.
                from bisect import bisect_right
                idx_harvested = bisect_right(state['plants'], threshold, lo=state['start_idx'])
                
                count = idx_harvested - state['start_idx']
                state['start_idx'] = idx_harvested
                return count
            return None

        # Using a list comprehension to trigger the function and filtering Nones
        # We import bisect_right inside handle, but for efficiency, 
        # let's redefine the logic slightly to avoid imports inside loops.
        return [handle(q) for q in qs]

    # Redefining handle without internal imports and using a class for state
    class State:
        def __init__(self):
            self.time = 0
            self.plants = []
            self.start_idx = 0
        
        def update(self, q):
            t, v = q
            if t == 1:
                self.plants.append(self.time)
                return None
            if t == 2:
                self.time += v
                return None
            if t == 3:
                threshold = self.time - v
                # Find index of first element > threshold
                import bisect
                idx = bisect.bisect_right(self.plants, threshold, lo=self.start_idx)
                count = idx - self.start_idx
                self.start_idx = idx
                return count

    s = State()
    results = [s.update(q) for q in queries]
    # Filter out None values and print
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == '__main__':
    solve()