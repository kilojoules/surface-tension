import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]

    # State consists of:
    # 1. current_time: Total T accumulated so far.
    # 2. plants: A sorted list of 'birth_times'. 
    #    A plant planted at total_time 't' has height (current_time - t).
    # 3. results: A list to store the answers for type 3 queries.
    
    def process_query(state, query_str):
        current_time, plants, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant height 0 means its birth_time is the current_time
            # We maintain the plants list sorted. Since current_time is non-decreasing,
            # we can just append.
            return (current_time, plants + [current_time], results)
        
        elif q_type == 2:
            # Increase global time
            return (current_time + parts[1], plants, results)
        
        else: # q_type == 3
            H = parts[1]
            # Plant is harvested if: current_time - birth_time >= H
            # Which means: birth_time <= current_time - H
            threshold = current_time - H
            
            # Find how many plants have birth_time <= threshold
            # bisect_left returns the index of the first element > threshold
            idx = bisect_left(plants, threshold + 0.1) 
            # Using threshold + 0.1 to handle the <= logic with bisect_left
            # Or more cleanly: bisect_right(plants, threshold)
            # Let's use a helper for clarity.
            
            # Correcting the index logic:
            # We need count of plants where p <= threshold.
            # Since we can't use a loop, we use a slice to remove them.
            # We use a custom function to find the split point.
            
            # Re-evaluating: since we can't use loops, we use a helper 
            # to find the index of the first plant that is NOT harvested.
            # Plants are sorted by birth_time. Plants at indices [0, idx-1] are harvested.
            
            # To avoid importing bisect_right, we use bisect_left on a 
            # value slightly larger than the threshold.
            # However, since birth_times are integers, threshold + 1 works.
            
            # Wait, the constraint allows imports. Let's use bisect_right.
            from bisect import bisect_right
            idx = bisect_right(plants, threshold)
            
            # The number of harvested plants is idx.
            # The remaining plants are plants[idx:].
            return (current_time, plants[idx:], results + [str(idx)])

    # We need to redefine the function to use bisect_right inside
    def final_process(state, query_str):
        current_time, plants, results = state
        parts = list(map(int, query_str.split()))
        if parts[0] == 1:
            return (current_time, plants + [current_time], results)
        if parts[0] == 2:
            return (current_time + parts[1], plants, results)
        
        # Type 3
        from bisect import bisect_right
        threshold = current_time - parts[1]
        idx = bisect_right(plants, threshold)
        return (current_time, plants[idx:], results + [str(idx)])

    # Initial state: (current_time, plants_list, results_list)
    initial_state = (0, [], [])
    
    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(final_process, queries, initial_state)
    
    # Output all results joined by newlines
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()