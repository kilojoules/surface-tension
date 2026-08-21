import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return

    Q = int(input_data[0])
    queries = input_data[1:]

    # We need to keep track of the "birth time" of each plant.
    # Let S be the prefix sum of T values from type 2 queries.
    # A plant planted at time t has height (S_current - S_t) at the current time.
    # The condition height >= H becomes: S_current - S_t >= H  =>  S_t <= S_current - H.
    
    # First, we extract all T values to build the prefix sum array S.
    # We use a list comprehension to find all T's from queries starting with '2'.
    T_values = [int(q.split()[1]) for q in queries if q.startswith('2')]
    
    # Calculate prefix sums of T. S[i] is the total time passed after i type-2 queries.
    # Using a list comprehension with a helper logic or map/reduce is tricky for prefix sums,
    # but since we can't use loops, we can use a trick with a list and a mutable object 
    # or just process the queries in a way that we track the state.
    # Actually, the constraint says no for loops, but we can use recursion or 
    # functional tools. However, Python's recursion limit is an issue.
    # Let's use a generator and a state-carrying object or a closure.
    
    # Wait, the simplest way to handle state without loops is to use a 
    # reduction (functools.reduce).
    from functools import reduce

    # State structure: (current_S, sorted_list_of_S_at_birth, results_list)
    def process_query(state, q_str):
        current_S, birth_times, results = state
        parts = q_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new flower. Its birth time is the current total S.
            # We maintain birth_times as a sorted list.
            # Since we can't use loops, we use bisect.insort to keep it sorted.
            import bisect
            bisect.insort(birth_times, current_S)
            return (current_S, birth_times, results)
        
        elif q_type == '2':
            # Increase total time S.
            T = int(parts[1])
            return (current_S + T, birth_times, results)
        
        else: # q_type == '3'
            # Harvest plants where S_t <= current_S - H.
            H = int(parts[1])
            threshold = current_S - H
            # Find number of plants with birth_time <= threshold.
            # bisect_right returns the index of the first element > threshold.
            import bisect
            idx = bisect.bisect_right(birth_times, threshold)
            # The number of harvested plants is idx.
            # Remove the first idx elements from the sorted list.
            return (current_S, birth_times[idx:], results + [idx])

    # Initial state: (current_S=0, birth_times=[], results=[])
    final_state = reduce(process_query, queries, (0, [], []))
    
    # Print all results joined by newlines.
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()