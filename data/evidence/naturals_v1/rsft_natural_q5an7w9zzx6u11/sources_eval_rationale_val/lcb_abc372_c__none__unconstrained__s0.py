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
    
    # Convert S to a list for mutability (though we treat it as state in reduce)
    S = list(S_str)
    
    # Initial count of "ABC"
    # We use a generator expression inside sum() to count occurrences
    initial_count = sum(1 for i in range(N - 2) if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C')
    
    # Prepare queries as a list of tuples (X, C)
    # Queries start from index 3 of input_data, every 2 elements is one query
    queries = [
        (int(input_data[i]), input_data[i+1]) 
        for i in range(3, len(input_data), 2)
    ]
    
    def check_abc(s, i):
        """Checks if 'ABC' starts at index i."""
        if 0 <= i <= N - 3:
            return 1 if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C' else 0
        return 0

    def process_query(state, query):
        current_S, current_count = state
        X, C = query
        idx = X - 1
        
        # 1. Subtract contributions of 'ABC' that involve the character at idx
        # A character at idx can be the 1st, 2nd, or 3rd char of 'ABC'
        loss = check_abc(current_S, idx - 2) + \
               check_abc(current_S, idx - 1) + \
               check_abc(current_S, idx)
        
        # 2. Update the character
        current_S[idx] = C
        
        # 3. Add contributions of 'ABC' that now involve the character at idx
        gain = check_abc(current_S, idx - 2) + \
               check_abc(current_S, idx - 1) + \
               check_abc(current_S, idx)
        
        return (current_S, current_count - loss + gain)

    # Use reduce to process all queries and collect the count after each step
    # Since reduce only returns the final state, we use a trick to store history
    # by wrapping the state in a list or using a custom accumulator.
    # However, to strictly avoid loops and maintain the "count" for each query,
    # we can use a helper function that appends the result to a list.
    
    results = []
    def reducer(state, query):
        new_state = process_query(state, query)
        results.append(new_state[1])
        return new_state

    reduce(reducer, queries, (S, initial_count))
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()