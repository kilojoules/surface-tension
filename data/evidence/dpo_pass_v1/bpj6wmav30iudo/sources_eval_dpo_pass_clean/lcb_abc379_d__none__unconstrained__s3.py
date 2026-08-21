import sys
from bisect import bisect_right

# Increase recursion depth for deep query lists
sys.setrecursionlimit(300000)

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We use a recursive function to process queries without for/while loops.
    # state: (current_time, plants_birth_times, query_index)
    # plants_birth_times is kept sorted.
    def process_queries(state):
        current_time, plants, idx = state
        
        # Base case: all queries processed
        if idx >= len(flat_queries):
            return
        
        q_type = flat_queries[idx]
        
        if q_type == '1':
            # Plant height 0 means it is born at the current_time
            # Since current_time is non-decreasing, plants remains sorted
            new_state = (current_time, plants + [current_time], idx + 1)
            process_queries(new_state)
            
        elif q_type == '2':
            # T is the next element in the flat list
            T = int(flat_queries[idx + 1])
            new_state = (current_time + T, plants, idx + 2)
            process_queries(new_state)
            
        elif q_type == '3':
            # H is the next element in the flat list
            H = int(flat_queries[idx + 1])
            # Plant is harvested if: current_time - birth_time >= H
            # birth_time <= current_time - H
            threshold = current_time - H
            
            # Find how many plants have birth_time <= threshold
            # bisect_right returns the index of the first element > threshold
            count_harvested = bisect_right(plants, threshold)
            
            # Output the result immediately
            sys.stdout.write(str(count_harvested) + '\n')
            
            # Remove harvested plants using slicing
            new_state = (current_time, plants[count_harvested:], idx + 2)
            process_queries(new_state)

    # Flatten the input into a list of strings to avoid loops
    flat_queries = input_data[1:]
    process_queries((0, [], 0))

if __name__ == "__main__":
    solve()