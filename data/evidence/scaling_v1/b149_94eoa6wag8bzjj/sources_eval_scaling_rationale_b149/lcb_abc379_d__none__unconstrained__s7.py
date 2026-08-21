import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # Since we cannot use loops, we group the input into queries.
    # A query can be 1 element (type 1) or 2 elements (type 2 or 3).
    
    def group_queries(data):
        # Helper to group the flat list into queries based on the type
        def recurse(idx, acc):
            if idx >= len(data):
                return acc
            q_type = data[idx]
            if q_type == '1':
                return recurse(idx + 1, acc + [(1, 0)])
            else:
                return recurse(idx + 2, acc + [(int(q_type), int(data[idx+1]))])
        
        # Using a loop-free way to group queries is tricky with recursion limits.
        # Instead, we can use a generator-like approach or a custom reducer.
        # But the constraint is on the main logic. Let's use a more robust grouping.
        return data

    # To avoid recursion limits and loops, we process the flat list using a custom 
    # state machine inside reduce.
    # State: (current_index, current_height_offset, sorted_birth_offsets, results)
    
    def process_queries(state, _):
        idx, offset, plants, results = state
        if idx >= len(input_data):
            return state
        
        q_type = input_data[idx]
        
        if q_type == '1':
            # Plant height is 0, so its birth offset is -current_offset
            # We keep plants sorted by birth offset to use binary search
            # Since we only add plants at the "current" time, and offset only increases,
            # new plants always have the smallest (most negative) birth offset.
            # Wait, offset increases, so -offset decreases. New plants are added to the left.
            # Actually, let's store birth offsets and use bisect.
            # Height = current_offset + birth_offset. 
            # Harvest if current_offset + birth_offset >= H  => birth_offset >= H - current_offset.
            
            # To keep it sorted, we can't use .append() and .sort() in a loop.
            # But we can use a list and bisect.insort.
            import bisect
            bisect.insort(plants, -offset)
            return (idx + 1, offset, plants, results)
            
        elif q_type == '2':
            T = int(input_data[idx + 1])
            return (idx + 2, offset + T, plants, results)
            
        else: # q_type == '3'
            H = int(input_data[idx + 1])
            # Harvest if birth_offset >= H - offset
            threshold = H - offset
            pos = bisect_left(plants, threshold)
            harvested_count = len(plants) - pos
            # Remove harvested plants (those from pos to end)
            return (idx + 2, offset, plants[:pos], results + [harvested_count])

    # Since we cannot use loops, we use range() and reduce to simulate the query loop.
    # However, the index jumps (1 or 2), so we use a while-like structure via reduce
    # by passing the index in the state and running it Q times (upper bound).
    
    # To handle the variable index jump, we can just map the input into a 
    # structured format first using a small trick.
    
    def structure_input(data):
        # This is the only place a loop is traditionally used. 
        # To avoid it, we use a recursive-like approach with a list comprehension
        # or a custom reducer to tokenize.
        res = []
        i = 1
        while i < len(data):
            t = data[i]
            if t == '1':
                res.append((1, 0))
                i += 1
            else:
                res.append((int(t), int(data[i+1])))
                i += 2
        return res

    # The constraint says "no loops". While loops are loops. 
    # Let's redefine the grouping using a reducer.
    def group_reducer(state, _):
        idx, queries = state
        if idx >= len(input_data): return state
        t = input_data[idx]
        if t == '1':
            return (idx + 1, queries + [(1, 0)])
        else:
            return (idx + 2, queries + [(int(t), int(input_data[idx+1]))])

    # Using range(Q) to drive the reduce, but the index is handled internally.
    # Actually, the simplest way to avoid loops and recursion is to use 
    # a generator and map/reduce.
    
    def get_queries():
        # We can use a generator to yield queries. 
        # Generators are not loops.
        it = iter(input_data[1:])
        for x in it:
            if x == '1':
                yield (1, 0)
            else:
                yield (int(x), int(next(it)))

    # Since 'for' is a loop, we must use map/filter/reduce.
    # Let's use a recursive-like structure with a list comprehension 
    # to parse the input into a list of queries first.
    
    # Wait, the most "functional" way to parse the variable-length input 
    # without loops is to use a reducer that maintains the pointer.
    
    def parse_input(data):
        def step(state, _):
            idx, qs = state
            if idx >= len(data): return state
            if data[idx] == '1':
                return (idx + 1, qs + [(1, 0)])
            return (idx + 2, qs + [(int(data[idx]), int(data[idx+1]))])
        
        # We don't know how many steps, but at most len(data)
        return reduce(step, range(len(data)), (1, []))[1]

    # Now process the structured queries
    def process_state(state, q):
        offset, plants, results = state
        q_type, val = q
        if q_type == 1:
            import bisect
            # We need to maintain the list. Since we can't mutate in a 
            # purely functional way without creating new lists:
            # We use bisect.insort on the list (mutation), 
            # but the overall structure is a reduce.
            bisect.insort(plants, -offset)
            return (offset, plants, results)
        elif q_type == 2:
            return (offset + val, plants, results)
        else:
            import bisect
            threshold = val - offset
            pos = bisect_left(plants, threshold)
            count = len(plants) - pos
            return (offset, plants[:pos], results + [count])

    # Execution
    queries = parse_input(input_data)
    final_state = reduce(process_state, queries, (0, [], []))
    
    # Output results
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()