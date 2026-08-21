import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    Q = int(input_data[1])
    S_str = input_data[2]
    queries = input_data[3:]
    
    # Convert S to a list for mutability (though we handle it via reduce)
    S = list(S_str)
    
    # Initial count of "ABC"
    # Using a list comprehension to find all occurrences and summing them
    initial_count = sum(1 for i in range(N - 2) if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    # Process queries using reduce
    # State: (current_S, current_count, results_list)
    # We process queries in chunks of 2 (X_i, C_i)
    
    def process_query(state, query_pair):
        curr_S, curr_count, results = state
        X = int(query_pair[0]) - 1 # 0-indexed
        C = query_pair[1]
        
        # We only need to check substrings that overlap with index X
        # These are substrings starting at X-2, X-1, and X
        # Ensure indices are within bounds [0, N-3]
        check_indices = [i for i in range(X - 2, X + 1) if 0 <= i <= N - 3]
        
        # Count "ABC" in the affected range before the change
        before = sum(1 for i in check_indices if curr_S[i] == 'A' and curr_S[i+1] == 'B' and curr_S[i+2] == 'C')
        
        # Apply the change
        curr_S[X] = C
        
        # Count "ABC" in the affected range after the change
        after = sum(1 for i in check_indices if curr_S[i] == 'A' and curr_S[i+1] == 'B' and curr_S[i+2] == 'C')
        
        # Update total count
        new_count = curr_count - before + after
        results.append(str(new_count))
        
        return (curr_S, new_count, results)

    # Group queries into pairs
    query_pairs = [queries[i:i+2] for i in range(0, len(queries), 2)]
    
    # Execute the reduction
    final_state = reduce(process_query, query_pairs, (S, initial_count, []))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()