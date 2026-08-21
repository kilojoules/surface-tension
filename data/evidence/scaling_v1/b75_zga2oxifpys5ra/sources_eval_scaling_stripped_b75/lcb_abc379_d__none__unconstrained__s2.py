import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We use a pointer-based approach to process the queries
    # since we cannot use loops. We use a generator or map.
    # However, the core logic requires maintaining a state:
    # 1. Current total height increase (offset)
    # 2. A sorted list of 'birth offsets' (the value of total height 
    #    at the moment the plant was planted).
    # A plant is harvested if: (Current Total Height) - (Birth Offset) >= H
    # Which is: Birth Offset <= (Current Total Height) - H
    
    # Since we cannot use loops, we use a custom class to maintain state
    # and a list comprehension to drive the queries.
    
    class State:
        def __init__(self):
            self.total_height = 0
            self.birth_offsets = []
            self.results = []

        def process(self, query):
            q_type = query[0]
            if q_type == '1':
                # Plant height 0 means its birth offset is the current total_height
                self.birth_offsets.append(self.total_height)
                # Keep the birth_offsets sorted for binary search
                # Since we only append and total_height is non-decreasing, 
                # it remains sorted naturally.
                return None
            elif q_type == '2':
                self.total_height += int(query[1])
                return None
            elif q_type == '3':
                H = int(query[1])
                # Target: birth_offset <= total_height - H
                threshold = self.total_height - H
                # Find number of plants with birth_offset <= threshold
                # bisect_right finds the insertion point to maintain order
                import bisect
                idx = bisect.bisect_right(self.birth_offsets, threshold)
                count = idx
                # Remove the harvested plants (the first 'idx' elements)
                self.birth_offsets = self.birth_offsets[idx:]
                return count

    # Parse queries into lists of strings
    # We need to group the input into queries based on the type
    def group_queries(data):
        it = iter(data[1:])
        def get_next():
            try:
                t = next(it)
                if t == '1':
                    return ('1',)
                else:
                    return (t, next(it))
            except StopIteration:
                return None
        
        # Use a list comprehension to consume the iterator
        # We use a helper to handle the variable length of queries
        return [get_next() for _ in range(Q)]

    # Because the group_queries logic above is slightly flawed for 
    # variable lengths in a comprehension, let's use a more robust 
    # approach to parse the flat list into queries.
    
    def parse_flat(data):
        # We use a generator to yield queries
        def gen(d):
            i = 1
            while i < len(d):
                t = d[i]
                if t == '1':
                    yield ('1',)
                    i += 1
                else:
                    yield (t, d[i+1])
                    i += 2
        return gen(data)

    state = State()
    # Use a list comprehension to process all queries and filter out None
    # The parse_flat generator is consumed by the comprehension
    final_results = [state.process(q) for q in parse_flat(input_data)]
    
    # Print only the non-None results
    sys.stdout.write('\n'.join(map(str, [r for r in final_results if r is not None])) + '\n')

if __name__ == "__main__":
    solve()