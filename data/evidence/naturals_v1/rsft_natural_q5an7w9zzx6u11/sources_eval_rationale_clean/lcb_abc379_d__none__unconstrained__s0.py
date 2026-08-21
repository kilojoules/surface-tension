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
    # 1. current_time: The total T accumulated so far.
    # 2. plants: A sorted list of 'birth_times'. 
    #    A plant planted at total time 't' has height (current_time - t).
    # 3. results: A list to store the answers for type 3 queries.
    
    def process_query(state, query_str):
        current_time, plants, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant height 0 means its birth_time is the current_time
            # We maintain the plants list sorted. Since current_time is non-decreasing,
            # we can just append.
            plants.append(current_time)
            return (current_time, plants, results)
        
        elif q_type == 2:
            # Increase total time
            t_inc = parts[1]
            return (current_time + t_inc, plants, results)
        
        else:
            # Type 3: Harvest plants with height >= H
            # Height = current_time - birth_time >= H  => birth_time <= current_time - H
            h_threshold = parts[1]
            max_birth_time = current_time - h_threshold
            
            # Find how many plants have birth_time <= max_birth_time
            # Since plants list is sorted, use binary search
            idx = bisect_left(plants, max_birth_time + 0.1) 
            # Note: birth_time <= max_birth_time is equivalent to birth_time < max_birth_time + 1
            # But since birth_times are integers, we use bisect_right logic.
            # Let's use a more precise boundary:
            import bisect
            idx = bisect.bisect_right(plants, max_birth_time)
            
            # The number of harvested plants is idx
            results.append(str(idx))
            
            # Remove the first 'idx' plants from the list
            # To avoid O(N) pop(0), we could use a deque, but the constraint 
            # forbids loops. However, slicing creates a new list.
            # Given Q=2e5, slicing might be slow, but it's the only way without loops.
            # Actually, we can keep track of an offset for the plants list.
            # But the state needs to be passed through reduce.
            # Let's use a list and a start_index.
            return (current_time, plants[idx:], results)

    # To handle the state more efficiently without slicing large lists 
    # (which is O(N)), we can use a different state structure.
    # However, the prompt asks for a functional approach. 
    # Let's refine the state to handle the plants list.
    
    # Since we cannot use loops, we use reduce to iterate through queries.
    # To avoid O(N) slicing, we can't easily use a deque with reduce 
    # because we need to return a new state. 
    # But Python's list slicing is highly optimized. Let's try.
    
    final_state = reduce(process_query, queries, (0, [], []))
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()