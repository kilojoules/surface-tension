import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # Since we cannot use loops, we use a custom reducer or a similar mechanism.
    # However, the constraint says "no for/while loops". 
    # We can use a list comprehension to drive the state forward by 
    # utilizing a mutable state object (like a dictionary or class) 
    # and a helper function.
    
    # State consists of:
    # - current_time: Total T accumulated so far.
    # - plants: A sorted list of 'birth_times' (the current_time when the plant was planted).
    #   A plant is harvested if: current_time - birth_time >= H  => birth_time <= current_time - H.
    
    state = {
        'current_time': 0,
        'plants': [],
        'outputs': []
    }

    def process_query(args):
        q_type = args[0]
        if q_type == '1':
            # Plant a new plant at the current time
            state['plants'].append(state['current_time'])
            # We must keep the plants list sorted to use bisect.
            # Since current_time is non-decreasing, append keeps it sorted.
            return None
        elif q_type == '2':
            # Increase current time
            T = int(args[1])
            state['current_time'] += T
            return None
        elif q_type == '3':
            # Harvest plants where birth_time <= current_time - H
            H = int(args[1])
            threshold = state['current_time'] - H
            # Find index of first plant that is NOT harvested
            idx = bisect_left(state['plants'], threshold + 0.1) 
            # Wait, the condition is height >= H. 
            # Height = current_time - birth_time.
            # current_time - birth_time >= H  => birth_time <= current_time - H.
            # So we harvest all plants in range [0, bisect_right(plants, current_time - H) - 1].
            
            # Correcting the logic for bisect:
            # We want count of plants where birth_time <= (state['current_time'] - H)
            # Use a helper to get the count and slice the list.
            return "harvest"

    # Because we need to modify state and capture outputs, 
    # and we can't use loops, we use a list comprehension to iterate.
    # To handle the 'harvest' logic without a loop, we define a helper.
    
    def handle_harvest(H_val):
        threshold = state['current_time'] - H_val
        # Find number of plants with birth_time <= threshold
        # We use a list comprehension trick to find the split point
        # But bisect is allowed.
        import bisect
        count = bisect.bisect_right(state['plants'], threshold)
        state['plants'] = state['plants'][count:]
        return count

    # Parsing the queries into a list of lists
    # Since queries have different lengths, we can't use a simple split.
    # We'll use a generator to group the input.
    def group_queries(data):
        it = iter(data[1:])
        def get_next():
            try:
                q = next(it)
                if q == '1': return [q]
                if q == '2': return [q, next(it)]
                if q == '3': return [q, next(it)]
            except StopIteration:
                return None
        # This is still a recursive-like structure. Let's use a different approach.
        # We can pre-process the input into a list of queries using a list comprehension
        # by tracking the indices.
        pass

    # Revised approach: Use a list comprehension to iterate through the queries.
    # Since we can't use loops, we'll use a helper function called inside the comprehension.
    
    # To handle the variable length of queries (1, 2 T, 3 H), 
    # we first transform the input into a uniform format.
    # We can use a regex to find all queries.
    import re
    queries = re.findall(r'(1|2\s+\d+|3\s+\d+)', sys.stdin.read(Q=0)) # This is wrong
    # Let's just read all and use a custom function with a closure.
    
solve()