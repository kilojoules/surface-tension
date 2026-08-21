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
    # Convert string to list for mutability (though we handle it via reduce)
    S = list(S_str)
    
    # Queries start from index 3, each query is (X, C)
    queries = []
    # Using a list comprehension to group queries into pairs
    # We use a range with a step of 2 to simulate the query pairs
    queries = [ (input_data[i], input_data[i+1]) for i in range(3, len(input_data), 2) ]

    # Initial count of "ABC"
    # We use a generator expression inside sum() to count occurrences
    initial_count = sum(1 for i in range(N - 2) if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')

    def process_query(state, query):
        current_S, current_count = state
        X_str, C = query
        idx = int(X_str) - 1
        
        # We only need to check substrings that overlap with the changed character
        # The character at idx can be the 1st, 2nd, or 3rd char of "ABC"
        # Indices to check as starting positions: idx-2, idx-1, idx
        check_indices = [i for i in range(idx - 2, idx + 1) if 0 <= i <= N - 3]
        
        # Count "ABC" at these positions before the change
        before = sum(1 for i in check_indices if current_S[i] == 'A' and current_S[i+1] == 'B' and current_S[i+2] == 'C')
        
        # Apply the change
        current_S[idx] = C
        
        # Count "ABC" at these positions after the change
        after = sum(1 for i in check_indices if current_S[i] == 'A' and current_S[i+1] == 'B' and current_S[i+2] == 'C')
        
        new_count = current_count - before + after
        
        # We print the result immediately. 
        # Since we cannot use a loop, we use sys.stdout.write
        sys.stdout.write(str(new_count) + '\n')
        
        return (current_S, new_count)

    # Use reduce to iterate through queries and maintain state
    reduce(process_query, queries, (S, initial_count))

if __name__ == "__main__":
    solve()