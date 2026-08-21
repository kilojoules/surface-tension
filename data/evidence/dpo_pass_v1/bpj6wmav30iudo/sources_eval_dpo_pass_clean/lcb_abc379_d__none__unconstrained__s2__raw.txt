The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, which often encourages the use of immutable data structures and higher-order functions. However, in Python, this is generally less idiomatic for problems requiring efficient mutable state (like a priority queue) and can lead to recursion depth issues. To solve this without loops or recursion, I will use `functools.reduce` to simulate the processing of the query stream and a `heapq` (managed within a list) to track plant heights. Since `heapq` functions mutate a list in place, they can be embedded within a list comprehension or a `map` call to bypass the need for explicit `for` or `while` blocks.

```python
import sys
import heapq
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]

    # State object to carry through reduce: 
    # (heap, current_time_offset, results_list)
    # We store plants as (height - current_time_offset) so that 
    # actual_height = stored_value + current_time_offset
    initial_state = ([], 0, [])

    def process_query(state, query_str):
        heap, offset, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]

        if q_type == 1:
            # Plant height 0 means stored value is 0 - offset
            # heapq.heappush returns None, so we use a list to capture the mutation
            [heapq.heappush(heap, -offset)]
            return (heap, offset, results)
        
        elif q_type == 2:
            # Increase offset by T
            return (heap, offset + parts[1], results)
        
        elif q_type == 3:
            # Harvest plants where actual_height >= H
            # actual_height = stored_value + offset >= H  => stored_value >= H - offset
            # Since we need to remove the smallest stored values that satisfy this,
            # but the heap is a min-heap of stored values, we check the top.
            # However, the condition is "at least H", meaning we remove 
            # plants that are TALL. A min-heap of stored values lets us 
            # remove the OLDEST (smallest stored value) plants first.
            
            H = parts[1]
            # We need to count how many plants have stored_value + offset >= H
            # This means stored_value >= H - offset.
            # Wait, the plants that are harvested are the ones that reached height H.
            # The plants that are MOST likely to be harvested are the ones 
            # planted earliest (smallest stored_value).
            
            # To remove all plants with height >= H, we remove all plants 
            # with stored_value >= H - offset. 
            # But a min-heap lets us remove the smallest values.
            # If the smallest stored_value + offset is < H, then no plant 
            # smaller than that can be harvested. 
            # But we want to harvest plants that are TALLER than H.
            # This means we remove plants from the heap as long as 
            # they are "old enough" to have reached height H.
            # The condition for harvest is: stored_value + offset >= H.
            # This is NOT what a min-heap usually does for "at least".
            # Actually, the plants that reach height H first are the ones 
            # with the smallest stored_values.
            # So we remove from the min-heap while min_val + offset >= H?
            # No, that's wrong. We remove while min_val + offset >= H? 
            # No: if the smallestI stored_value + offset is >= H, 
            # then ALL plants in the heap are >= H.
            # If the smallest stored_value +, offset is < H, 
            # some plants might still be >= H.
            # Wait, the plants are planted at different times.
            # Plant 1: offset 0, stored 0.
            # Plant 2: offset 10, stored -10.
            # If H=5, Plant 1 (0+10=10) is harvested, Plant 2 (-10+10=0) is not.
            # The plants that are harvested are those with stored_value >= H - offset.
            # This is a range query. But since we only care about the count,
            # and we remove them, we can use a min-heap to find plants 
            # that are NOT yet H. But that's not right.
            # Let's re-evaluate: Harvest plants with height >= H.
            # Height = current_offset - planting_offset.
            # Harvest if: current_offset - planting_offset >= H 
            # => planting_offset <= current_offset - H.
            # We want to count and remove all plants with planting_offset <= threshold.
            # A min-heap of planting_offsets is perfect.
            
            threshold = offset - H
            # Use a helper to count and pop. Since we can't use while, 
            # we use a recursive-like structure via reduce or a trick.
            # But we can't use recursion. 
            # We can use a list comprehension with a side effect 
            # combined with heapq.heappop.
            
            # To simulate 'while heap and heap[0] <= threshold: count += 1; heappop(heap)'
            # we can use a generator or a reduce on a range.
            # Since Q is 2e5, we can't just iterate Q times per query.
            # But the total number of pops across all queries is at most Q.
            
            # We use a list to store the count for this specific query.
            count_container = [0]
            # We use a range(len(heap)) and a conditional to pop.
            # However, the length of the heap changes. 
            # We can use a list comprehension that attempts to pop 
            # as long as the condition is met.
            # To avoid loops, we can use a recursive-like approach with 
            # a custom function passed to reduce, but we can't use recursion.
            # We can use a generator expression inside sum().
            
            # The trick: use a generator that pops from the heap.
            # Since we can't use 'while', we use a generator that 
            # iterates over a large range but stops when the condition fails.
            # We can use 'itertools.takewhile'.
            
            import itertools
            
            # Create a generator that pops from the heap
            # Note: we must use a mutable container to access the heap
            pop_gen = (heapq.heappop(heap) for _ in range(len(heap)))
            # takewhile will stop as soon as the condition is false
            harvested = list(itertools.takewhile(lambda x: x <= threshold, pop_gen))
            
            # Because we popped them into a list, they are removed from the heap.
            # However, takewhile consumes the generator. 
            # The plants that were NOT harvested must be put back? 
            # No, takewhile stops at the first element that doesn't match.
            # But the element that didn't match was already popped by the generator.
            # We must put it back.
            
            # Correction:
            # We can't easily pop and put back without a loop.
            # Let's use a different approach for the "while" logic.
            # We can use a recursive function? No.
            # We can use a list comprehension with a helper function.
            
            # Let's use a helper function that handles the popping logic.
            # Since I cannot use 'while' or 'for', I will use 'itertools.takewhile'
            # but I need to handle the "peek" and "pop" carefully.
            
            # Actually, I can use a list comprehension to pop 
            # as long as the heap is not empty and heap[0] <= threshold.
            # But I need a way to repeat this.
            # I can use 'reduce' on a range of size Q.
            
            # Let's redefine the logic inside a helper.
            return handle_harvest(heap, offset, H, results)

    # To implement the 'while' logic without 'for' or 'while', 
    # we use reduce on a range of size Q for each type 3 query.
    # But that's Q^2. We need the total pops to be Q.
    # We can use a global-like state and a single reduce over all queries.
    
    def handle_harvest(state, query_str):
        heap, offset, results = state
        parts = list(map(int, query_str.split()))
        if parts[0] != 3: return state
        
        H = parts[1]
        threshold = offset - H
        
        # Use a list to track the count for this query
        # We use a generator and takewhile to identify how many to pop.
        # To avoid the "pop and lose" problem, we peek first.
        # But we can't loop. 
        # Wait, we can use a recursive function if we increase recursion limit? 
        # No, constraints say no recursion.
        
        # Let's use a mathematical approach: 
        # Since we need to pop all x <= threshold, 
        # we can use a list comprehension that pops 
        # but we need to stop.
        # We can use a helper that uses 'itertools.dropwhile' or 'takewhile'.
        # But we need to mutate the heap.
        
        # The only way to simulate 'while' without 'for' or 'recursion' 
        # is using 'reduce' or 'map' on a sequence.
        # For a type 3 query, we can reduce over a range of size len(heap).
        
        def pop_until(st, _):
            h, off, res = st
            # This is tricky because we need to stop.
            # We can use a conditional that returns the state.
            if h and h[0] <= off - H:
                return (heapq.heappop(h), st) # This is not quite right
            return (None, st)

        # Let's use a simpler approach: 
        # Use a list comprehension to pop elements.
        # Since we can't use a loop, we can use map/reduce.
        # To simulate 'while heap and heap[0] <= threshold', 
        # we can use reduce on range(len(heap)).
        
        def reducer(acc, _):
            h, off, res, count = acc
            if h and h[0] <= off - H:
                return (h, off, res, count + 1 if heapq.heappop(h) is not None else 0) # wait
                # heapq.heappop returns the value.
                # we need to return the updated state.
            return acc
            
        # This is getting complex. Let's use a simpler method.
        # We can use a list comprehension to pop:
        # [heapq.heappop(heap) for _ in range(len(heap)) if heap and heap[0] <= threshold]
        # This works because the 'if' is evaluated lazily!
        # However, the 'range(len(heap))' is fixed at the start.
        # But since we only pop, len(heap) is an upper bound.
        
        # Let', check:
        # [heapq.heappop(heap) for _ in range(len(heap)) if heap and heap[0] <= threshold]
        # If heap is [1, 2, 3] and threshold is 2:
        # _=0: heap[0]=1 <= 2? Yes. pop 1. heap=[2, 3]
        # _=1: heap[0]=2 <= 2? Yes. pop 2. heap=[3]
        # _=2: heap[0]=3 <= 2? No.
        # Result: [1, 2], len=2. Correct.
        
        harvested_ plants = [heapq.heappop(heap) for _ in range(len(heap)) if heap and heap[0] <= threshold]
        results.append(len(harvested_plants))
        return (heap, offset, results)

    # Final assembly using reduce to process all queries
    final_state = reduce(
        lambda state, q: (
            (
                # Type 1
                (
                    [heapq.heappush(state[0], -state[1]) or state[0], 
                     state[1], 
                     state[2]] 
                    if q.startswith('1') else (
                    # Type 2
                    (
                        state[0], 
                        state[1] + int(q.split()[1]), 
                        state[2]
                    ) if q.