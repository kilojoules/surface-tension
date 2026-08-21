import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We need to track the height of plants.
    # Let 'current_time' be the total T accumulated from type 2 queries.
    # A plant planted at 'current_time' with height 0 has a "birth_time" of 'current_time'.
    # Its height at any future time is: (current_time_now - birth_time).
    # The condition height >= H becomes: (current_time_now - birth_time) >= H
    # Which is equivalent to: birth_time <= (current_time_now - H).
    
    # We store birth_times of all existing plants in a sorted list.
    # Since we only add plants (type 1) and remove the smallest birth_times (type 3),
    # and birth_times are added in non-decreasing order, the list remains sorted.
    
    # Using a list and bisect for range queries. 
    # However, removing elements from the front of a list is O(N).
    # To keep it O(1) or O(log N), we use a pointer or collections.deque.
    # But we can't use deque with bisect. 
    # Instead, we maintain the birth_times in a list and use a pointer 'start_idx'.
    
    # Process queries without loops using a generator or map.
    # Since we have state (current_time, birth_times, start_idx), 
    # we can use a custom class or a closure to maintain state.
    
    class State:
        def __init__(self):
            self.current_time = 0
            self.birth_times = []
            self.start_idx = 0
            
        def process(self, q_type, *args):
            if q_type == '1':
                self.birth_times.append(self.current_time)
                return None
            elif q_type == '2':
                self.current_time += int(args[0])
                return None
            elif q_type == '3':
                H = int(args[0])
                threshold = self.current_time - H
                # Find how many plants have birth_time <= threshold
                # We search in the slice birth_times[start_idx:]
                # The index returned by bisect is relative to the whole list.
                idx = bisect_left(self.birth_times, threshold + 1, lo=self.start_idx)
                count = idx - self.start_idx
                self.start_idx = idx
                return count

    state = State()
    
    # Parse the flat input list into (type, arg) pairs
    # This is tricky because queries have different lengths.
    # We use a helper to group them.
    def group_queries(data):
        it = iter(data)
        for item in it:
            if item == '1':
                yield ('1',)
            elif item == '2':
                yield ('2', next(it))
            elif item == '3':
                yield ('3', next(it))

    # Execute and filter out None values
    results = [state.process(*q) for q in group_queries(queries)]
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()