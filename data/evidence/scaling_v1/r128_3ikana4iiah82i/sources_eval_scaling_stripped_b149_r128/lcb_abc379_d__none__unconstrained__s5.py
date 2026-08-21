import sys
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to track the "birth time" of plants relative to a global clock.
    # Let 'current_time' be the sum of all T from type 2 queries.
    # A plant planted at 'current_time' has an initial height of 0.
    # Its height at any future time is: (future_time - birth_time).
    # The condition height >= H becomes: (future_time - birth_time) >= H
    # Which rearranges to: birth_time <= (future_time - H).
    
    # We will store the birth_times of all active plants in a sorted list.
    # Since we only add plants (type 1) and remove the smallest birth_times (type 3),
    # we can use a sorted list and a pointer (or slicing) to track active plants.
    
    # However, since we cannot use external libraries like SortedList, 
    # and we need to remove elements from the start, we can use a 
    # combination of a list and a pointer.
    
    # Parse queries into a list of tuples
    # This is a bit tricky because queries have different lengths.
    # We'll use a generator to yield the queries.
    
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
    
    # To handle the "harvest" efficiently:
    # We maintain a sorted list of birth_times.
    # Type 1: Append current_time to the list. (Since current_time is non-decreasing, 
    # the list remains sorted).
    # Type 3: Find how many birth_times are <= (current_time - H).
    # All those plants are removed.
    
    # Since we need to remove from the front, we can use a deque or 
    # simply keep track of the index of the first active plant.
    # But wait, the plants are added at the end and removed from the front.
    # This is a perfect use case for a pointer.
    
    # Let's redefine:
    # plants = list of birth_times
    # start_idx = index of the first plant that hasn't been harvested.
    
    # We can't use a loop to process queries, so we'll use a custom 
    # reduction or a recursive-like structure. 
    # Actually, the simplest way to "loop" without 'for' or 'while' 
    # is using map() or list comprehensions, but we have state.
    # We can use a mutable object (like a dictionary) inside a list comprehension.
    
    state = {
        'current_time': 0,
        'plants': [],
        'start_idx': 0,
        'results': []
    }
    
    def process(q):
        q_type, val = q
        if q_type == 1:
            state['plants'].append(state['current_time'])
            return None
        elif q_type == 2:
            state['current_time'] += val
            return None
        else:
            # Type 3: Harvest plants with birth_time <= current_time - H
            threshold = state['current_time'] - val
            # Find the number of plants from start_idx onwards that are <= threshold
            # We use bisect_left on the slice plants[start_idx:]
            # But slicing is O(N). We should use the full list and adjust the index.
            
            # The plants list is sorted. We find the first index i where plants[i] > threshold.
            # The number of harvested plants is i - start_idx.
            idx = bisect_left(state['plants'], threshold + 1, lo=state['start_idx'])
            count = idx - state['start_idx']
            state['start_idx'] = idx
            return count

    # Use map to apply the process function to all queries
    # Filter out None values to get the results of type 3 queries
    ans = [res for res in map(process, queries) if res is not None]
    
    sys.stdout.write('\n'.join(map(str, ans)) + '\n')

if __name__ == "__main__":
    solve()