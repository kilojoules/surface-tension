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
    # A plant planted at time 't' has height (current_time - t).
    # The condition height >= H becomes: current_time - t >= H  =>  t <= current_time - H.
    
    # We store the planting times of all existing plants in a sorted list.
    # Since we only add plants (type 1) and remove them (type 3), 
    # and the condition t <= threshold is a prefix of the sorted list,
    # we can use a sorted list and track the index of the first plant NOT harvested.
    
    # However, plants are added at different times. To keep the list sorted,
    # we can't use a simple list if we add plants arbitrarily.
    # Wait, the plants are always added at the 'current' time.
    # Since current_time is non-decreasing, the planting times are added in 
    # non-decreasing order. A simple list suffices.
    
    # We use a list to store planting times and a pointer/index to track 
    # which plants have been harvested. But plants are harvested based on 
    # a threshold that changes. 
    # Actually, the condition is t <= current_time - H.
    # As current_time increases, the threshold (current_time - H) doesn't 
    # necessarily increase. But we only care about plants currently in the pots.
    
    # Let's maintain a list of planting times of active plants.
    # Since we only add to the end, the list is always sorted.
    # When harvesting, we remove all plants from the start of the list 
    # whose planting time is <= current_time - H.
    
    # Using a deque for efficient popping from the left.
    from collections import deque
    
    # We can't use a loop to process queries because of the constraint.
    # We use a generator or map.
    
    # State: (deque_of_planting_times, current_time)
    # We need a way to update the state and return the count for type 3.
    
    # Since we need to maintain state across queries, we use a class or a closure.
    class State:
        def __init__(self):
            self.plants = deque()
            self.current_time = 0
            
        def process(self, query_str):
            parts = query_str.split()
            q_type = parts[0]
            
            if q_type == '1':
                self.plants.append(self.current_time)
                return None
            elif q_type == '2':
                self.current_time += int(parts[1])
                return None
            else: # q_type == '3'
                H = int(parts[1])
                threshold = self.current_time - H
                # Count how many plants have planting_time <= threshold
                # Since plants is sorted, we find the number of elements <= threshold.
                # We can use bisect_left on the deque (converted to list) 
                # but popping from the left is the goal.
                
                # To avoid loops, we can use a list and a pointer.
                # But the problem says we must remove them.
                # Let's use a list and a 'start_index'.
                return threshold

    # Revised approach: 
    # Use a list for planting times and a variable for the current start index.
    # Since we can't use loops, we'll use a helper function with a mutable state object.
    
    class Tracker:
        def __init__(self):
            self.planting_times = []
            self.current_time = 0
            self.start_idx = 0
            
        def handle(self, q_str):
            p = q_str.split()
            if p[0] == '1':
                self.planting_times.append(self.current_time)
                return None
            if p[0] == '2':
                self.current_time += int(p[1])
                return None
            # Type 3
            H = int(p[1])
            threshold = self.current_time - H
            # Find how many plants from start_idx onwards are <= threshold
            # We use bisect_left on the slice planting_times[start_idx:]
            # But slicing is O(N). We use the whole list and adjust the index.
            idx = bisect_left(self.planting_times, threshold + 1, lo=self.start_idx)
            count = idx - self.start_idx
            self.start_idx = idx
            return count

    tracker = Tracker()
    results = map(tracker.handle, queries)
    
    # Filter out None values and print
    sys.stdout.write('\n'.join(map(str, filter(lambda x: x is not None, results))) + '\n')

if __name__ == "__main__":
    solve()