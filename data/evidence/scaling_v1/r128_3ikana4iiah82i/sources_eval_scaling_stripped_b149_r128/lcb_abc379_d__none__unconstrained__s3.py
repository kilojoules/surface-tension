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
    # A plant planted at 'current_time' has an initial height of 0.
    # Its height at any future time is (future_time - planting_time).
    # The condition height >= H becomes: (current_time - planting_time) >= H
    # Which rearranges to: planting_time <= (current_time - H)
    
    # We store the planting times of all active plants in a sorted list.
    # Since we only add plants (type 1) and remove them (type 3),
    # and the condition is a prefix of the sorted planting times,
    # we can use a sorted list and bisect to find the number of plants to harvest.
    
    # However, Python's list.pop(0) is O(N). To keep it O(log N) or O(1) amortized,
    # we can use a deque or simply track the index of the first active plant.
    # But since we need to remove elements from the start, a pointer is sufficient.
    
    # Let's use a list to store planting times and a pointer 'start_idx'.
    # Note: We use a list comprehension to parse queries to avoid explicit for loops.
    # But the logic requires state updates. We can use a generator or a custom class.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.planting_times = []
            self.start_idx = 0
            self.results = []

        def process(self, q_str):
            parts = q_str.split()
            q_type = parts[0]
            
            if q_type == '1':
                self.planting_times.append(self.current_time)
            elif q_type == '2':
                self.current_time += int(parts[1])
            elif q_type == '3':
                H = int(parts[1])
                # Condition: planting_time <= current_time - H
                threshold = self.current_time - H
                # Find number of plants with planting_time <= threshold
                # We search in the slice [start_idx:]
                # Using bisect_right to find the number of elements <= threshold
                # We adjust the search range by adding start_idx to the result
                idx = bisect_left(self.planting_times, threshold, lo=self.start_idx)
                
                # bisect_left returns the first index where element is >= threshold.
                # We want elements <= threshold. 
                # Let's use bisect_right for "less than or equal to".
                from bisect import bisect_right
                idx = bisect_right(self.planting_times, threshold, lo=self.start_idx)
                
                count = idx - self.start_idx
                self.results.append(str(count))
                self.start_idx = idx

    # To avoid explicit loops, we can use map() or a generator.
    # Since we need to maintain state, we can use a helper function with a closure
    # or a class. The constraint says no explicit for/while loops.
    
    state = State()
    # Use map to apply the process method to every query
    list(map(state.process, queries))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(state.results) + '\n')

if __name__ == "__main__":
    solve()