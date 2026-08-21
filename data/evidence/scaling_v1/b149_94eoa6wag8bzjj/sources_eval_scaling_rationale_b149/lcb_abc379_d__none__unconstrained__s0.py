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
    # The input format varies (1, 2 T, 3 H), so we use a generator-like 
    # approach or a mapping to structure the data.
    
    # To handle the variable length of queries without a loop, 
    # we can pre-process the flat list into a list of queries.
    # However, since we can't use loops to parse, we use a recursive-like 
    # structure via a custom function and reduce, but the prompt forbids recursion.
    # Actually, we can parse the flat list by iterating through it using a 
    # state-based reduce or by using a clever slice/map if we knew the lengths.
    # Given the constraints, the most reliable way to parse variable length 
    # input without loops is to use an iterator.
    
    it = iter(input_data[1:])
    
    def get_queries(iterator):
        # Since we can't use loops or recursion, we can't easily build a list
        # of variable-length queries. But wait, we can use a list comprehension
        # if we have a way to track the index. 
        # Actually, the simplest way to handle this is to realize that 
        # we can use a while-loop replacement: reduce over a range.
        pass

    # Let's redefine: we will use reduce over the range of Q, 
    # but we need to track the current position in the input_data list.
    
    # State: (current_input_idx, current_height_offset, sorted_birth_offsets, results)
    # birth_offset = current_height_offset - plant_height_at_birth
    # A plant is harvested if: current_height_offset + birth_offset >= H
    # Which means: birth_offset >= H - current_height_offset
    
    def process_queries(state, _):
        idx, offset, plants, results = state
        q_type = input_data[idx]
        
        if q_type == '1':
            # Plant height 0. Birth offset = 0 - offset = -offset
            # We maintain plants list sorted.
            new_plants = plants + [-offset] 
            # Note: plants + [x] is O(N). To keep it sorted and efficient, 
            # we should use bisect.insort, but that mutates.
            # However, since we can't use loops, we must use functions.
            # To keep it O(log N), we use bisect.insort.
            import bisect
            bisect.insort(plants, -offset)
            return (idx + 1, offset, plants, results)
        
        elif q_type == '2':
            T = int(input_data[idx + 1])
            return (idx + 1 + 1, offset + T, plants, results)
        
        else: # q_type == '3'
            H = int(input_data[idx + 1])
            # Harvest if: offset + birth_offset >= H  => birth_offset >= H - offset
            threshold = H - offset
            pos = bisect_left(plants, threshold)
            harvested_count = len(plants) - pos
            # Remove harvested plants (those from pos to end)
            # We use slice assignment to mutate the list in place
            del plants[pos:]
            results.append(str(harvested_count))
            return (idx + 2, offset, plants, results)

    # We use reduce to simulate the loop over Q queries.
    # We start idx at 1 because input_data[0] is Q.
    final_state = reduce(process_queries, range(Q), (1, 0, [], []))
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(final_state[3]) + '\n')

if __name__ == "__main__":
    solve()