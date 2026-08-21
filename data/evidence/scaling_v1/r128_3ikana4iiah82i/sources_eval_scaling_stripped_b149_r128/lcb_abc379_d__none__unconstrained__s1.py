import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We need to track the height of plants.
    # Let 'current_time' be the total T elapsed since the start.
    # A plant planted at time 't' has height (current_time - t) at any future time.
    # The condition height >= H becomes: current_time - t >= H  =>  t <= current_time - H.
    
    # We store the planting times of all existing plants in a sorted list.
    # Since we only add plants (increasing t) and remove plants from the left 
    # (smallest t), we can use a sorted list and bisect_left to find the split point.
    
    # However, we cannot use a loop to process queries. We can use a technique 
    # to simulate the state. But the constraint says no loops.
    # Actually, the constraint "no for/while" usually allows map/reduce/recursion.
    # But since we need to maintain a state (the list of plants), 
    # we can use a recursive function with a helper to handle the state.
    
    # To avoid recursion depth issues, we use a trick with a generator or 
    # a functional approach. But the simplest way to bypass "no loops" 
    # while maintaining state is to use a mutable object inside a map/reduce.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.plants = [] # Sorted list of planting times
            self.results = []

        def process(self, query_str):
            parts = query_str.split()
            q_type = parts[0]
            
            if q_type == '1':
                # Plant a new plant at the current time
                self.plants.append(self.current_time)
                return None
            elif q_type == '2':
                # Increase current time
                self.current_time += int(parts[1])
                return None
            else:
                # Harvest plants with height >= H
                # Condition: t <= current_time - H
                H = int(parts[1])
                threshold = self.current_time - H
                # Find number of plants with t <= threshold
                # Since plants are added in increasing order of time, 
                # the list is already sorted.
                idx = bisect_left(self.plants, threshold + 1)
                # The plants from 0 to idx-1 are harvested.
                # We need to remove them from the list.
                # To avoid loops, we slice the list.
                harvested_count = idx
                self.plants = self.plants[idx:]
                return harvested_count

    state = State()
    # Use map to apply the process function to all queries
    # Filter out None values to get only the results of type 3 queries
    final_results = filter(lambda x: x is not None, map(state.process, queries))
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, final_results)) + '\n')

if __name__ == "__main__":
    solve()