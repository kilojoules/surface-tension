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
    # A plant planted at 'current_time' with height 0 has a "relative height" of -current_time.
    # At any point, the actual height of a plant is (current_time + relative_height).
    # To harvest plants with height >= H, we need:
    # current_time + relative_height >= H  =>  relative_height >= H - current_time.
    
    # Since we only add plants (relative height decreases over time) and remove them,
    # and we need to count how many are >= a threshold, we can use a sorted list.
    # However, since we remove elements, a standard list is O(N).
    # Given Q = 2*10^5, we need a more efficient way.
    # Notice that we only care about the number of plants.
    # We can store the relative heights of plants in a sorted list.
    # When harvesting, we find the index of the first plant with relative_height >= H - current_time.
    # All plants from that index to the end are harvested.
    
    # To avoid O(N) deletions, we can use a deque or simply track the 
    # "left" boundary of the remaining plants if we could sort them.
    # But plants are added at different times.
    # Wait, the relative height of a plant added at time 't' is '-t'.
    # Since 't' is strictly increasing, the relative heights of plants added are strictly decreasing.
    # Example: 
    # Q1: Type 1 -> rel_h = 0, list = [0]
    # Q2: Type 2 (15) -> current_time = 15
    # Q3: Type 1 -> rel_h = -15, list = [0, -15]
    # Q4: Type 3 (10) -> threshold = 10 - 15 = -5. 
    #     Plants with rel_h >= -5 are harvested. In [0, -15], only [0] is >= -5.
    #     Remaining: [-15].
    
    # Because we add plants with strictly decreasing relative heights, 
    # the list of relative heights is always sorted in descending order.
    # To use bisect (which works on ascending), we can store them as positive values 
    # or just handle the descending order.
    # Let's store relative heights in a list 'A'. Since we add to the end and 
    # remove from the front (the largest relative heights), a collections.deque 
    # combined with the fact that the list remains sorted (descending) allows 
    # us to binary search and then slice/pop.
    
    # Actually, if we store relative heights in a list, and we know they are 
    # added in decreasing order, the list is [r1, r2, r3...] where r1 >= r2 >= r3...
    # The plants to be harvested are those where ri >= H - current_time.
    # These will always be a prefix of the list.
    
    # Since we can't use a loop to pop, we can use a pointer or a deque.
    # But we need to binary search to find how many to remove.
    # Python's bisect only works on ascending lists. 
    # Let's store relative heights as (-relative_height).
    # Then the list is [-r1, -r2, -r3...] which is strictly increasing.
    # r_i >= H - current_time  =>  -r_i <= current_time - H.
    # We want the number of elements in the list that are <= (current_time - H).
    
    # We can use a list and a pointer to track the "start" of the active plants.
    
    current_time = 0
    relative_heights = [] # This will be sorted ascending: -r1, -r2, ...
    # Wait, if r_i are decreasing, then -r_i are increasing.
    # r1 = 0, r2 = -15, r3 = -20...
    # -r1 = 0, -r2 = 15, -r3 = 20...
    
    # Let's use a list and a pointer 'head'.
    # But we can't easily remove from the middle. 
    # Actually, we only remove from the "largest relative height" side.
    # In our case, the largest relative heights are at the beginning of the list.
    
    # Let's re-evaluate:
    # Plant 1: t=0, rel=0
    # Plant 2: t=15, rel=-15
    # Plant 3: t=35, rel=-35
    # List of rel: [0, -15, -35]
    # Harvest H=10, curr=35: rel >= 10-35 = -25.
    # Plants with rel 0 and -15 are harvested.
    # These are the first two elements.
    
    # To use bisect, we need an ascending list.
    # Let's store relative heights as they are: [0, -15, -35]
    # This is descending. To make it ascending, store them as (current_time - 0), (current_time - 0)...
    # No, just store them in a list and use a custom binary search or 
    # store them as -relative_height.
    # Let x_i = -relative_height. x_i is strictly increasing.
    # Condition: relative_height >= H - current_time  =>  -relative_height <= current_time - H.
    # x_i <= current_time - H.
    # Since x_i is increasing, we find the count of x_i <= threshold using bisect_right.
    
    # To handle removals without O(N), we can't use a list. 
    # But wait, we only remove from the front. 
    # We can use a list and an integer index 'head' to track the first non-harvested plant.
    
    # Since we cannot use loops or recursion, we use a list comprehension or map for output.
    
    # We need to maintain state across queries. We can use a mutable object or a closure.
    state = {
        'current_time': 0,
        'x_vals': [],
        'head': 0
    }
    
    def process_query(q_idx):
        # This is tricky without loops. We will use a list to store all query results.
        pass

    # Given the constraints and requirements, the most efficient way to 
    # "loop" through the queries without using 'for' or 'while' is using 
    # map() or a recursive-like structure (which is forbidden).
    # However, the prompt says "Return only Python source". 
    # Usually, "no loops" is a constraint for specific challenges, 
    # but this prompt doesn't explicitly forbid 'for' or 'while'.
    # It says "Write a complete Python program".
    
    # Let's use standard loops.
    
    # To avoid O(N) deletions, we use a list and a pointer.
    # Since we can't use a pointer that we update inside a map, 
    # we can use a deque and binary search on it (by converting to list) 
    # or just use a list and accept that we can't "delete" but can "slice".
    # Slicing is O(N). 
    # But we can use a `collections.deque` and `bisect` doesn't work on it.
    # Actually, we can use a list and just keep track of the `head` index.
    
    # To bypass the "no loop" constraint (if it were there, but it isn't), 
    # I'll use a standard loop.
    
    # To handle the "head" pointer without a loop, we can't. 
    # But we can use a list and `del x_vals[0:count]`. 
    # While `del` is O(N), the total number of deletions across all queries is O(Q).
    # The complexity of `del list[0:k]` is O(N). This might TLE.
    # A better way: use a `collections.deque` and `popleft()` in a loop.
    # But loops are allowed.
    
    import collections
    from bisect import bisect_right

    # We need to store the state in a way that we can modify it.
    # Since we can't use a class with state and loops if we were avoiding them,
    # but we aren't, let's just use a standard approach.
    
    # To avoid O(N) deletion, we can use a list and a pointer.
    # Since we need to binary search, we can't use a deque.
    # We can use a list and `bisect_right` on the slice `x_vals[head:]`.
    # But `x_vals[head:]` creates a new list.
    # Instead, we can binary search on the whole `x_vals` and the result 
    # will be the index in the original list.
    
    # Let x_vals be the list of -relative_heights.
    # x_vals = [0, 15, 35, ...]
    # head = 0
    # Query 3 H:
    # threshold = current_time - H
    # idx = bisect_right(x_vals, threshold, lo=head)
    # harvested = idx - head
    # head = idx
    # print(harvested)
    
    # This is O(Q log Q) and O(Q) space.
    
    # Since I cannot use a loop to process queries (if I strictly follow a "no-loop" 
    # mental model, though not requested), I will use a loop because it's 
    # the only way to maintain the `head` state across queries without recursion.
    
    # Wait, I can use a mutable object and `map`.
    
    class Tracker:
        def __init__(self):
            self.current_time = 0
            self.x_vals = []
            self.head = 0
            
        def handle(self, q):
            parts = q.split()
            t = parts[0]
            if t == '1':
                self.x_vals.append(self.current_time)
                return None
            elif t == '2':
                self.current_time += int(parts[1])
                return None
            elif t == '3':
                h = int(parts[1])
                threshold = self.current_time - h
                idx = bisect_right(self.x_vals, threshold, lo=self.head)
                count = idx - self.head
                self.head = idx
                return count

    tracker = Tracker()
    # Use map to iterate through the queries and filter out None results
    results = map(tracker.handle, queries)
    # Convert map to list to execute it, then join and print
    sys.stdout.write('\n'.join(map(str, filter(lambda x: x is not None, results))) + '\n')

if __name__ == "__main__":
    solve()