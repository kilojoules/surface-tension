import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    S_init = input_data[2]
    
    # Queries are pairs of (X_i, C_i)
    queries = [
        (int(input_data[i]), input_data[i+1]) 
        for i in range(3, len(input_data), 2)
    ]
    
    # Initial count of "ABC"
    # We use a list comprehension to find all occurrences and sum them
    initial_count = sum(1 for i in range(N - 2) if S_init[i:i+3] == "ABC")
    
    # Convert string to list for mutability (though we treat it via reduce)
    S_list = list(S_init)
    
    def process_query(state, query):
        current_S, current_count = state
        X, C = query
        idx = X - 1
        
        # Identify the range of indices that could be affected by changing S[idx]
        # A change at idx can affect substrings starting at idx-2, idx-1, and idx.
        start = max(0, idx - 2)
        end = min(N - 3, idx)
        
        # Calculate how many "ABC"s are in the affected range before the change
        # We use a generator expression inside sum()
        before = sum(1 for i in range(start, end + 1) if "".join(current_S[i:i+3]) == "ABC")
        
        # Update the character
        current_S[idx] = C
        
        # Calculate how many "ABC"s are in the affected range after the change
        after = sum(1 for i in range(start, end + 1) if "".join(current_S[i:i+3]) == "ABC")
        
        new_count = current_count - before + after
        return (current_S, new_count)

    # Use reduce to iterate through queries and maintain state
    # We need to capture the count at each step, so we wrap the reduce logic
    # to store results in a list. Since reduce only returns the final state,
    # we use a trick: we store the results in a list passed along in the state.
    
    def reduce_with_history(state, query):
        (current_S, current_count, history) = state
        # Logic from process_query
        X, C = query
        idx = X - 1
        start, end = max(0, idx - 2), min(N - 3, idx)
        before = sum(1 for i in range(start, end + 1) if "".join(current_S[i:i+3]) == "ABC")
        current_S[idx] = C
        after = sum(1 for i in range(start, end + 1) if "".join(current_S[i:i+3]) == "ABC")
        new_count = current_count - before + after
        history.append(str(new_count))
        return (current_S, new_count, history)

    # Initial state: (S_list, initial_count, history_list)
    final_state = reduce(reduce_with_history, queries, (S_list, initial_count, []))
    
    # Print all results joined by newlines
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()