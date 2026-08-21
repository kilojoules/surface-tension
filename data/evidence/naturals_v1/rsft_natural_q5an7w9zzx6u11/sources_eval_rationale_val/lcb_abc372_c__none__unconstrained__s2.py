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
    
    # Initial count of "ABC"
    # Using a list comprehension and sum to avoid explicit for loop
    initial_count = sum(1 for i in range(N - 2) if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    # Prepare queries as a list of tuples (X, C)
    # Using slice and zip to pair X_i and C_i
    queries = zip(
        map(int, input_data[3::2]), 
        input_data[4::2]
    )
    
    def check_abc(idx):
        """Checks if 'ABC' starts at index idx. Returns 1 if true, 0 otherwise."""
        if 0 <= idx <= N - 3:
            return 1 if S[idx] == 'A' and S[idx+1] == 'B' and S[idx+2] == 'C' else 0
        return 0

    def process_query(state, query):
        curr_count, _ = state
        x_i, c_i = query
        idx = x_i - 1 # 1-indexed to 0-indexed
        
        # A change at index idx can only affect "ABC" substrings starting at 
        # idx-2, idx-1, and idx.
        # Subtract the contributions of these positions before the change.
        before = check_abc(idx - 2) + check_abc(idx - 1) + check_abc(idx)
        
        # Apply the change
        S[idx] = c_i
        
        # Add the contributions of these positions after the change.
        after = check_abc(idx - 2) + check_abc(idx - 1) + check_abc(idx)
        
        new_count = curr_count - before + after
        # We return the new count and the updated S (though S is mutated in place)
        # We print the result immediately using a side-effect in the reduce function
        # However, since we need to print Q lines, we can collect results in a list.
        return (new_count, S)

    # Use reduce to process queries. 
    # Since we need to output the count after each query, 
    # we'll use a list to store results and map/reduce to fill it.
    
    # To avoid a loop for printing, we can use a list to capture the counts
    # and then join them with newlines.
    
    results = []
    
    def reducer(state, query):
        res = process_query(state, query)
        results.append(res[0])
        return res

    reduce(reducer, queries, (initial_count, S))
    
    # Output all results joined by newline
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()