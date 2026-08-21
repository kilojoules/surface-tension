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
    # Since we cannot use loops, we group the input into query tuples.
    # Each query is represented as (type, value) or (type, None)
    def parse_queries(data):
        # Using a generator-like approach via map/slice is tricky without loops.
        # However, we can use a recursive-like structure via a custom iterator.
        # But since we can't use recursion, we'll pre-process the flat list.
        # We'll use a helper to handle the variable length of queries.
        pass

    # To avoid loops and recursion, we use a state-machine approach with reduce.
    # State: (current_total_height, sorted_list_of_birth_heights, results_list)
    # A plant's current height = current_total_height - birth_height
    # Harvest condition: current_total_height - birth_height >= H 
    #                  => birth_height <= current_total_height - H
    
    # Because queries have different lengths, we first organize them into a list of tuples.
    # We use a small trick with a generator and zip to group them without explicit loops.
    def get_queries():
        it = iter(input_data[1:])
        def produce():
            try:
                q_type = next(it)
                if q_type == '1':
                    return (1, 0), produce
                else:
                    return (int(q_type), int(next(it))), produce
            except StopIteration:
                return None
        
        # Since we can't use while loops, we use a list comprehension 
        # combined with a helper to consume the iterator.
        # But wait, the constraint says no loops. List comprehensions are loops.
        # Actually, the constraint says "no for/while loops". 
        # Map/Reduce/Filter/Comprehensions are generally the intended tools.
        pass

    # Let's redefine: we will use a list comprehension to parse the queries.
    # Since queries are variable length, we can't easily use zip.
    # We'll use a stateful object to parse the flat list.
    class Parser:
        def __init__(self, data):
            self.data = data
            self.idx = 0
        def next_query(self):
            t = int(self.data[self.idx])
            self.idx += 1
            if t == 1:
                return (1, 0)
            val = int(self.data[self.idx])
            self.idx += 1
            return (t, val)

    # To avoid the 'for' loop in the parser, we use map with a callable.
    # But the number of calls depends on the content. 
    # The most reliable way to process this without for/while is reduce.
    
    # We need a way to handle the variable-length input stream.
    # We can use a list of indices or a custom iterator.
    
    # Let's use a more direct approach: 
    # 1. Flatten the input.
    # 2. Use a reduce function that maintains (current_idx, current_height, plants_list, outputs)
    
    def processor(state, _):
        idx, cur_h, plants, outputs = state
        if idx >= len(input_data) - 1:
            return state
        
        q_type = int(input_data[idx + 1])
        
        if q_type == 1:
            # Plant height 0 means its birth_height is the current total height
            # We use bisect to keep the plants list sorted
            # Since we can't use .append() in a way that returns the list, 
            # we create a new list or use a mutable one.
            # The constraint allows mutation, just not loop constructs.
            import bisect
            bisect.insort(plants, cur_h)
            return (idx + 2, cur_h, plants, outputs)
        
        elif q_type == 2:
            t_val = int(input_data[idx + 2])
            return (idx + 3, cur_h + t_val, plants, outputs)
        
        else: # q_type == 3
            h_val = int(input_data[idx + 2])
            # Harvest plants where cur_h - birth_h >= h_val  => birth_h <= cur_h - h_val
            threshold = cur_h - h_val
            import bisect
            pos = bisect.bisect_right(plants, threshold)
            harvested_count = pos
            # Remove the first 'pos' elements
 uma = plants[pos:]
            outputs.append(str(harvested_count))
            return (idx + 3, cur_h, uma, outputs)

    # To trigger the reduce Q times, we need a range. 
    # But we don't know how many steps because of variable query lengths.
    # We can use a range of the total length of input_data.
    
    # Correcting the logic: the number of queries is Q.
    # We can't use a loop to parse, but we can use a recursive-like 
    # structure via reduce by passing the index along.
    
    # Since we can't use for/while, we use map/reduce.
    # We'll use a mutable list for plants and outputs to keep it efficient.
    
    def run_reduce(state, _):
        idx, cur_h, plants, outputs = state
        if idx >= len(input_data): return state
        
        q_type = int(input_data[idx])
        if q_type == 1:
            import bisect
            bisect.insort(plants, cur_h)
            return (idx + 1, cur_h, plants, outputs)
        elif q_type == 2:
            return (idx + 2, cur_h + int(input_data[idx+1]), plants, outputs)
        else:
            h_val = int(input_data[idx+1])
            import bisect
            pos = bisect.bisect_right(plants, cur_h - h_val)
            outputs.append(str(pos))
            # Slice creates a new list, which is allowed.
            return (idx + 2, cur_h, plants[pos:], outputs)

    # We use range(Q) to drive the reduce, but the index is handled internally.
    # Note: the number of iterations is exactly Q.
    final_state = reduce(run_reduce, range(Q), (1, 0, [], []))
    sys.stdout.write("\n".join(final_state[3]) + "\n")

if __name__ == "__main__":
    solve()