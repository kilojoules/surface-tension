import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and maintain state.
    # State: (current_height_offset, sorted_list_of_planting_times, outputs)
    # A plant planted at time 't' has height: current_height_#offset - t
    # Condition height >= H becomes: current_height_offset - t >= H  => t <= current_height_offset - H
    
    # Parse queries into a list of tuples
    # Since queries have different lengths, we use a generator to yield them
    def get_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, 0)
                elif q_type == '2':
                    yield (2, int(next(it)))
                elif q_type == '3':
                    yield (3, int(next(it)))
            except StopIteration:
                break

    queries = list(get_queries(input_data))

    def process(state, query):
        offset, plants, outputs = state
        q_type, val = query
        
        if q_type == 1:
            # Plant height 0 means its "birth offset" is the current total offset
            # We keep plants sorted by their birth offset to use binary search
            # Since we always add the current offset, and offset is non-decreasing,
            # the list remains sorted.
            plants.append(offset)
            return (offset, plants, outputs)
        
        elif q_type == 2:
            # Increase height of all plants by T
            return (offset + val, plants, outputs)
        
        else: # q_type == 3
            # Harvest plants where height >= H
            # height = offset - birth_offset >= H  => birth_offset <= offset - H
            threshold = offset - val
            # Find index of first plant with birth_offset > threshold
            idx = bisect_left(plants, threshold + 1)
            harvested_count = idx
            # Remove harvested plants (those from 0 to idx-1)
            # Using slice assignment to modify the list in place
            del plants[:idx]
            outputs.append(str(harvested_count))
            return (offset, plants, outputs)

    # Initial state: (offset, plants_list, output_list)
    initial_state = (0, [], [])
    final_state = reduce(process, queries, initial_state)
    
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()