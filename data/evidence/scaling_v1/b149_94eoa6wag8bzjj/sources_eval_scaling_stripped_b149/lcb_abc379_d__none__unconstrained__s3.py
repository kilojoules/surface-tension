import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_height_offset tracks the total T added by type 2 queries.
    # When a plant is added at time t, its "birth height" is -current_height_offset.
    # Its actual height at any time is: birth_height + current_height_offset.
    # A plant is harvested if: birth_height + current_height_offset >= H
    # Which is: birth_height >= H - current_height_offset.
    
    current_height_offset = 0
    # Sorted list of birth heights of existing plants.
    # Since we need to remove elements and maintain sorted order, 
    # and Q is 2*10^5, a simple list with bisect and slice is O(Q^2) worst case.
    # However, we can use a SortedList from a library, but since we must use 
    # standard Python, we use a list and accept that slice deletion is the bottleneck.
    # For the given constraints, we use a list and bisect.
    plants = []
    
    # To avoid loops, we use a generator to process queries.
    # We use a helper function to maintain state via a closure or a class.
    class State:
        def __init__(self):
            self.offset = 0
            self.plants = []
            
        def process(self, q_str):
            parts = q_str.split()
            q_type = parts[0]
            
            if q_type == '1':
                # Plant height 0 means birth_height = -self.offset
                # Use bisect to keep plants list sorted
                import bisect
                bisect.insort(self.plants, -self.offset)
                return None
            
            elif q_type == '2':
                self.offset += int(parts[1])
                return None
            
            elif q_type == '3':
                H = int(parts[1])
                threshold = H - self.offset
                # Find first plant with birth_height >= threshold
                idx = bisect_left(self.plants, threshold)
                num_harvested = len(self.plants) - idx
                # Remove harvested plants
                self.plants = self.plants[:idx]
                return num_harvested

    state = State()
    # Use map to apply the process function to all queries
    results = map(state.process, queries)
    # Filter out None values and print
    sys.stdout.write('\n'.join(map(str, filter(lambda x: x is not None, results))) + '\n')

if __name__ == "__main__":
    solve()