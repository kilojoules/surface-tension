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
    # Convert string to list for mutability
    S = list(S_str)
    
    # Prepare queries as pairs of (index, char)
    # X_i is 1-indexed, convert to 0-indexed
    queries = [
        (int(input_data[i]), input_data[i+1]) 
        for i in range(3, len(input_data), 2)
    ]
    
    # Initial count of "ABC"
    # Use a generator expression inside sum() to count occurrences
    initial_count = sum(
        1 for i in range(N - 2) 
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C'
    )
    
    def process_query(state, query):
        current_S, current_count = state
        idx_1based, new_char = query
        idx = idx_1based - 1
        
        # We only need to check substrings that overlap with the changed index.
        # The affected indices for the start of "ABC" are idx-2, idx-1, and idx.
        # We filter these to stay within bounds [0, N-3].
        check_indices = [i for i in range(idx - 2, idx + 1) if 0 <= i <= N - 3]
        
        # Subtract occurrences that existed before the change
        count_before = sum(
            1 for i in check_indices 
            if current_S[i] == 'A' and current_S[i+1] == 'B' and current_S[i+2] == 'C'
        )
        
        # Apply the change
        current_S[idx] = new_char
        
        # Add occurrences that exist after the change
        count_after = sum(
            1 for i in check_indices 
            if current_S[i] == 'A' and current_S[i+1] == 'B' and current_S[i+2] == 'C'
        )
        
        new_count = current_count - count_before + count_after
        # We print the result immediately using a list comprehension trick or just print
        # Since we cannot use loops, we use sys.stdout.write
        sys.stdout.write(str(new_count) + '\n')
        
        return (current_S, new_count)

    # Use reduce to iterate through queries without a for/while loop
    reduce(process_query, queries, (S, initial_count))

if __name__ == "__main__":
    solve()