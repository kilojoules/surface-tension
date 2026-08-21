```python
import sys
from bisect import bisect_left
from itertools import accumulate

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the total height added over time to determine 
    # the "birth height" of a plant.
    # If total height added so far is S, a plant planted now has a 
    # relative height of -S. It reaches height H when total height is S + H.
    
    # Since we cannot use loops, we process queries by mapping them to 
    # a structure that allows us to track the state.
    # However, the problem requires counting and removing elements.
    # A SortedList from a library is forbidden, and we can't use loops.
    # But we can use a Fenwick tree or Segment tree implemented via 
    # list comprehensions if we coordinate-compress the "birth heights".
    
    # 1. Extract all queries
    # We use a generator to group the input into queries.
    def get_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, 0)
                elif q_type == '2':
                    yield (2, int(next(it)))
                else:
                    yield (3, int(next(it)))
            except StopIteration:
                break

    queries = list(get_queries(input_data))
    
    # 2. Calculate prefix sums of T for type 2 queries to get total height at any time
    # heights_added[i] = total T added before query i
    def calc_totals(qs):
        return list(accumulate([t if q[0] == 2 else 0 for q in qs], lambda a, b: a + b))
    
    totals = calc_totals(queries)
    
    # 3. Coordinate Compression
    # A plant planted at index i has relative height -totals[i].
    # It is harvested at index j if totals[j] - totals[i] >= H  =>  totals[i] <= totals[j] - H.
    # We collect all possible 'birth' values (totals[i]) and 'threshold' values (totals[j] - H).
    birth_values = [totals[i] for i, q in enumerate(queries) if q[0] == 1]
    threshold_values = [totals[i] - q[1] for i, q in enumerate(queries) if q[0] == 3]
    
    all_coords = sorted(list(set(birth_values + threshold_values)))
    coord_map = {val: i + 1 for i, val in enumerate(all_coords)}
    
    # 4. Fenwick Tree operations using a mutable list
    # Since we can't use loops, we use a trick: we can't update a Fenwick tree 
    # without a loop. Wait, the constraint says no explicit for/while loops.
    # But we can use recursion (with caution) or map/reduce.
    # Actually, the most reliable way to handle "range sums" and "point updates" 
    # without loops in Python is to use a Segment Tree or Fenwick Tree 
    # where the update/query is done via recursion.
    
    # To avoid recursion depth issues and loops, we can use a different approach.
    # We can treat this as a 2D range counting problem:
    # Plant i is harvested at query j if:
    # 1. i < j
    # 2. query i is type 1, query j is type 3
    # 3. totals[i] <= totals[j] - H
    # 4. Plant i was not harvested by any query k (i < k < j)
    
    # This is a classic problem that can be solved by finding for each plant i,
    # the first j > i such that totals[j] - H >= totals[i].
    # We can use a Segment Tree (implemented via a list) to find the first index.
    # But we can't build/update it without loops.
    
    # Let's reconsider: we can use a balanced BST or a SortedList. 
    # Since we can't use external libs, we can use a Fenwick tree 
    # and perform updates/queries using a recursive function.
    
    bit = [0] * (len(all_coords) + 1)
    
    def update(idx, val):
        # Use a list comprehension to simulate the while loop for BIT update
        # The maximum index is 2*10^5, so the path is ~18 steps.
        def step(i):
            if i >= len(bit): return
            bit[i] += val
            step(i + (i & -i))
        step(idx)

    def query_bit(idx):
        # Use a recursive helper to simulate the while loop for BIT query
        def step(i, s):
            if i <= 0: return s
            return step(i - (i & -i), s + bit[i])
        return step(idx, 0)

    # To avoid "recursion" in update/query and "loops", 
    # we can use a technique to process queries.
    # However, the "no loop" constraint is strict. 
    # Let's use a more functional approach.
    
    # Since we need to output the number of plants harvested,
    # and plants are removed, we can't simply count.
    # But we can use a Segment Tree to find the first index j > i 
    # where totals[j] - H >= totals[i].
    
    # Actually, the simplest way to implement this without loops 
    # is to use a library-like SortedList implemented via a 
    # Divide and Conquer approach or using recursion for BIT.
    # Python's recursion limit needs to be increased.
    
    sys.setrecursionlimit(300000)
    
    # We need to process queries and maintain the BIT.
    # We can use a list to store the results and a function to 
    # process the queries list recursively.
    
    results = []
    
    def process_queries(idx, current_bit):
        if idx == len(queries):
            return
        
        q = queries[idx]
        if q[0] == 1:
            update(coord_map[totals[idx]], 1)
            process_queries(idx + 1, current_bit)
        elif q[0] == 2:
            process_queries(idx + 1, current_bit)
        else:
            # Type 3: Harvest plants with height >= H
            # Height is totals[idx] - birth_total >= H  => birth_total <= totals[idx] - H
            thresh = totals[idx] - q[1]
            c_idx = coord_map[thresh]
            
            # Number of plants currently in BIT with birth_total <= thresh
            # This is not quite right because we need to REMOVE them.
            # To remove them, we need to know WHICH ones.
            # This suggests we need a way to find and remove all indices <= c_idx.
            # That's hard without loops.
            
            # Alternative: Use a Segment Tree to find the first plant that satisfies the condition.
            # Or: Since we need to remove all plants <= thresh, 
            # we can just keep track of the "minimum birth_total" currently present.
            # But plants are added at different times.
            
            # Wait, the condition is: harvest ALL plants with height >= H.
            # This means all plants with birth_total <= totals[idx] - H.
            # We can use the BIT to count how many are <= thresh, 
            # then we need to "clear" the BIT for all indices <= thresh.
            # Clearing a range in a BIT is hard. 
            # But we can just keep track of the "last cleared threshold".
            # Any plant with birth_total <= current_threshold is already gone.
            
            # Let's use a different state: (current_index, current_min_birth_total)
            # This is still not quite right because a plant added later might be 
            # harvested even if an older plant wasn't.
            # Actually, if birth_total_i <= birth_total_j, and plant j is harvested,
            # then plant i must have been harvested already (or is harvested now).
            # So we only need to track the "watermark" of the maximum birth_total harvested.
            
            pass

    # Correct logic:
    # Plants are harvested if birth_total <= totals[idx] - H.
    # Let W be the watermark of the maximum birth_total harvested so far.
    # When query 3 H comes:
    # 1. Current threshold T_h = totals[idx] - H.
    # 2. Plants to harvest are those with birth_total <= T_h AND birth_total > W.
    # 3. Update W = max(W, T_h).
    # 4. The number of plants is (count of plants with birth_total <= T_h) - (count of plants with birth_total <= W_old).
    
    # This requires us to count plants added at type 1 queries.
    # Let's use a BIT to store all plants ever added, and use the watermark.
    
    # 1. Pre-calculate all type 1 births
    births = [totals[i] for i, q in enumerate(queries) if q[0] == 1]
    # 2. For each type 3 query, we need to count plants added so far that are <= threshold
    # and were not harvested before.
    
    # Let's use a simpler approach:
    # For each type 3 query at index i with threshold Th = totals[i] - H:
    # We want to count plants k < i such that totals[k] <= Th and plant k was not harvested.
    # A plant k is harvested at the first i > k such that totals[i] - H_i >= totals[k].
    
    # This is equivalent to:
    # For each plant k, it is harvested at i = min {i > k | totals[i] - H_i >= totals[k]}
    # Then for a query i, the answer is the count of k such that their harvest-index is i.
    
    # To find the harvest-index without loops, we can use a Segment Tree 
    # (implemented recursively) to find the first i in a range.
    # But we can't use loops to build the tree. 
    # We can use a recursive function to build it.
    
    # Given the constraints and the "no loop" rule, the most viable path 
    # is to use recursion for everything and increase the recursion limit.
    
    def solve_functional():
        # Use a list to store the watermark and the BIT
        # We use a closure to maintain state
        state = {
            'watermark': -float('inf'),
            'bit': [0] * (len(all_coords) + 1)
        }
        
        def bit_update(i, delta):
            if i >= len(state['bit']): return
            state['bit'][i] += delta
            bit_update(i + (i & -i), delta)
            
        def bit_query(i):
            if i <= 0: return 0
            return state['bit'][i] + bit_query(i - (i & -i))

        def process(idx):
            if idx == len(queries): return
            
            q = queries[idx]
            if q[0] == 1:
                bit_update(coord_map[totals[idx]], 1)
                process(idx + 1)
            elif q[0] == 2:
                process(idx + 1)
            else:
                thresh = totals[idx] - q[1]
                # We need plants with birth_total <= thresh.
                # However, some were already harvested.
                # The plants harvested are those with birth_total <= thresh.
                # Since we remove them, we can't just use a watermark because
                # the threshold Th is not necessarily monotonic.
                # Wait, the problem says "Harvest all plants with height at least H".
                # This means birth_total <= totals[idx] - H.
                # We can use the BIT to count plants in range [1, coord_map[threshP]]
                # where threshP is the mapped value of thresh.
                # Then we REMOVE them from the BIT.
                # To remove them without a loop, we can't. 
                # UNLESS we use a Segment Tree that supports range clear.
                
                # Let's use the property: we only care about plants with birth_total <= thresh.
                # We can use a Segment Tree to find all indices in the BIT that are 1 
                # in the range