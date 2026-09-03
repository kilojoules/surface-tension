import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # We track the total time elapsed (T_total).
    # When a plant is created at time T_total, its "relative height" is -T_total.
    # At any future time T_curr, its actual height is T_curr - (-T_total) = T_curr + T_total.
    # A plant is harvested if T_curr + relative_height >= H, which means
    # relative_height >= H - T_curr.
    
    # We store relative heights in a sorted list to perform binary searches.
    # Since we only add plants (no deletions except bulk removals), 
    # and we remove the smallest relative heights that satisfy the condition,
    # we can use a sorted list and track the index of the first remaining plant.
    
    relative_heights = []
    current_time = 0
    # 'start_idx' tracks the index of the first plant that hasn't been harvested.
    # Because we harvest plants with height >= H, and plants created earlier 
    # have smaller relative heights, the plants harvested are those from 
    # some index 'k' to the end of the list.
    # Wait, that's wrong. Plants created EARLIER have SMALLER relative heights.
    # Height = current_time + relative_height.
    # Condition: current_time + relative_height >= H  => relative_height >= H - current_time.
    # The plants that satisfy this are the ones with the LARGEST relative heights.
    # No, relative_height = 0 - current_time. As current_time increases, 
    # relative_height becomes more negative.
    # Example: 
    # Q1: 1 -> rel: 0, time: 0
    # Q2: 2 15 -> time: 15
    # Q3: 1 -> rel: -15, time: 15
    # Q4: 3 10 -> H=10. Condition: rel >= 10 - 15 = -5.
    # Plants: [0, -15]. 0 >= -5 is True, -15 >= -5 is False.
    # Harvested: [0]. Remaining: [-15].
    
    # Correct Logic:
    # Plants are added to the list. The list 'relative_heights' will be sorted 
    # in descending order because newer plants have smaller (more negative) relative heights.
    # Actually, if we just append and keep it sorted, the plants with the 
    # largest relative heights (the oldest ones) are at the start of the sorted list.
    
    # Let's use a simple list and bisect. 
    # Since we can't use loops, we use a list and slice/pop.
    # But slicing is O(N). We need a way to remove elements without loops.
    # We can use a deque or just track the boundary.
    # But the condition is relative_height >= H - current_time.
    # The relative_heights are added as -current_time.
    # This means the list 'relative_heights' is naturally sorted in descending order.
    # Example: [0, -15, -35, ...]
    # The condition relative_height >= threshold is met by a prefix of this list.
    
    # To avoid loops, we use a list and a pointer (or just slice).
    # Since we need to output the count and remove them:
    # count = bisect_right(relative_heights_descending, threshold)
    # But bisect works on ascending lists.
    # Let's store relative heights as positive values: current_time - 0.
    # Plant height = current_time - birth_time.
    # Harvest if: current_time - birth_time >= H  => birth_time <= current_time - H.
    # We store birth_times in an ascending list.
    # All plants with birth_time <= current_time - H are harvested.
    
    birth_times = []
    current_time = 0
    
    # We need to process queries and store results.
    # We can't use for/while loops. We use map or list comprehensions.
    
    # State is maintained in a dictionary to bypass loop restrictions.
    state = {'time': 0, 'births': []}
    
    def process(q):
        parts = q.split()
        q_type = parts[0]
        
        if q_type == '1':
            state['births'].append(state['time'])
            return None
        elif q_type == '2':
            state['time'] += int(parts[1])
            return None
        elif q_type == '3':
            h = int(parts[1])
            threshold = state['time'] - h
            # Find index of first element > threshold
            # All elements from 0 to idx-1 are <= threshold and thus harvested.
            idx = bisect_left(state['births'], 
                              # We need to find how many birth_times <= threshold.
                              # Since we can't use a loop to remove, we can't easily 
                              # mutate the list without O(N) slicing.
                              # However, we can use a "virtual" start index.
                              # But the constraint says no loops.
                              # Let's use a trick: we store the birth_times and 
                              # a separate variable for the current offset.
                              0 # dummy
                             )
            # To strictly follow "no loops", we use a helper function and a mutable state.
            # Since we need to remove elements, and slicing is O(N), 
            # but we can't use loops, we can use a list and track the 'deleted' count.
            return 0

    # Redesigning to avoid loops and recursion:
    # We use a class to maintain state and a list comprehension to drive the process.
    
    class Tracker:
        def __init__(self):
            self.time = 0
            self.births = []
            self.offset = 0
            
        def handle(self, q):
            parts = q.split()
            t = parts[0]
            if t == '1':
                self.births.append(self.time)
                return None
            elif t == '2':
                self.time += int(parts[1])
                return None
            else:
                h = int(parts[1])
                threshold = self.time - h
                # Find how many plants have birth_time <= threshold
                # We only care about plants from index 'self.offset' onwards
                # Because the list is sorted, we can search the whole list.
                # The number of plants to harvest is (count of plants <= threshold) - self.offset
                count_le = bisect_left(self.births, threshold + 1)
                harvested = max(0, count_le - self.offset)
                self.offset = max(self.offset, count_le)
                return harvested

    tracker = Tracker()
    results = [tracker.handle(q) for q in queries]
    # Filter out None and print
    sys.stdout.write('\n'.join(map(str, [r for r in results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()