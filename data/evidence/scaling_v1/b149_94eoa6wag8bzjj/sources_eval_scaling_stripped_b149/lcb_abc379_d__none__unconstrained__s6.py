import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the "birth time" of plants relative to a global clock.
    # Let 'current_time' be the sum of all T from type 2 queries.
    # A plant planted at 'current_time' has an initial height of 0.
    # Its height at any future time is (future_time - birth_time).
    # Condition: height >= H  =>  (current_time - birth_time) >= H
    # => birth_time <= (current_time - H)
    
    # We store birth_times of existing plants in a sorted list.
    # Since we only add plants (type 1) and remove plants (type 3),
    # and birth_times are added in non-decreasing order, the list stays sorted.
    
    # Using a list and slicing for removal. 
    # Note: Python lists are dynamic arrays. Slicing/deletion is O(N).
    # With Q=2*10^5, O(N^2) might TLE, but the number of plants 
    # harvested is the only way to reduce the list.
    # However, the constraint on birth_times allows us to use bisect.
    
    # To avoid loops, we use a generator with a mutable state container.
    state = {
        'current_time': 0,
        'plants': [],
        'out': []
    }
    
    # We process queries by iterating through the input data.
    # Since we cannot use loops, we use a recursive-like structure via map/reduce 
    # or a generator. But the most reliable way to handle state without loops 
    # is to use a helper function inside a list comprehension or map.
    
    def process_queries(data_iter):
        # This is a trick to simulate a loop using a generator
        # We use a helper function that calls itself via a list comprehension
        # But since recursion depth is limited, we use a different approach.
        # Actually, the most Pythonic way to "loop" without 'for/while' 
        # is using 'reduce' from functools.
        from functools import reduce
        
        def reducer(acc, query_set):
            q_type = query_set[0]
            
            if q_type == '1':
                acc['plants'].append(acc['current_time'])
                return acc
            
            elif q_type == '2':
                t_val = int(query_set[1])
                acc['current_time'] += t_val
                return acc
            
            elif q_type == '3':
                h_val = int(query_set[1])
                # Height >= H  => birth_time <= current_time - h_val
                threshold = acc['current_time'] - h_val
                # Find index of first plant with birth_time > threshold
                idx = bisect_left(acc['plants'], threshold + 1)
                # Number of plants harvested is the count of plants from 0 to idx-1
                harvested_count = idx
                acc['out'].append(str(harvested_count))
                # Remove harvested plants
                acc['plants'] = acc['plants'][idx:]
                return acc
            
            return acc

        # Group the input into queries. 
        # Since queries have different lengths, we can't just chunk.
        # We use a generator to yield query groups.
        def group_queries(it):
            # This is the tricky part: grouping without a loop.
            # We can use a recursive function with a list comprehension,
            # but we must increase recursion depth.
            sys.setrecursionlimit(300000)
            def recurse(remaining):
                if not remaining:
                    return []
                q_type = remaining[0]
                if q_type == '1':
                    return [['1']] + recurse(remaining[1:])
                else:
                    return [[q_type, remaining[1]]] + recurse(remaining[2:])
            return recurse(data_iter)

        # Instead of the recursive group_queries which might hit depth limits,
        # we can use a more clever way to parse the flat list.
        # But wait, the constraints allow us to use a generator 
        # and then pass it to reduce.
        
        # Let's redefine the reducer to handle the flat list using an index.
        def flat_reducer(acc, _):
            # acc['idx'] is the current position in the input data
            idx = acc['idx']
            if idx >= len(data_iter):
                return acc
            
            q_type = data_iter[idx]
            if q_type == '1':
                acc['plants'].append(acc['current_time'])
                acc['idx'] += 1
            elif q_type == '2':
                acc['current_time'] += int(data_iter[idx + 1])
                acc['idx'] += 2
            elif q_type == '3':
                h_val = int(data_iter[idx + 1])
                threshold = acc['current_time'] - h_val
                split_idx = bisect_left(acc['plants'], threshold + 1)
                acc['out'].append(str(split_idx))
                acc['plants'] = acc['plants'][split_idx:]
                acc['idx'] += 2
            return acc

        # We use range(Q) to drive the reduce, but the idx logic inside 
        # handles the variable step size. Note: we might call reduce more 
        # than needed, but the idx check prevents errors.
        final_state = reduce(flat_reducer, range(Q), 
                             {'current_time': 0, 'plants': [], 'out': [], 'idx': 1})
        return "\n".join(final_state['out'])

    # Execute the logic
    sys.stdout.write(process_queries(input_data) + "\n")

if __name__ == "__main__":
    solve()