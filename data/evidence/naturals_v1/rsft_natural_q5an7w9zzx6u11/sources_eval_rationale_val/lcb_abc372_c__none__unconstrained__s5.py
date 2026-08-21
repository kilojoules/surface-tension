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
    
    # Queries start from index 3, each query takes 2 elements (X, C)
    queries = [
        (int(input_data[i]), input_data[i+1]) 
        for i in range(3, len(input_data), 2)
    ]
    
    # Initial count of "ABC"
    initial_count = S_str.count("ABC")
    
    def check_abc(s, i):
        """Checks if an 'ABC' starts at index i."""
        if 0 <= i <= N - 3:
            return 1 if (s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C') else 0
        return 0

    def process_query(state, query):
        current_s, current_count = state
        x, c = query
        idx = x - 1 # 1-indexed to 0-indexed
        
        # We only need to check indices that could be affected by the change at idx.
        # These are indices i such that the substring [i, i+2] covers idx.
        # i can be idx-2, idx-1, or idx.
        
        # Subtract contributions of existing ABCs in the affected range
        old_contribution = sum([check_abc(current_s, i) for i in range(idx - 2, idx + 1)])
        
        # Update the character
        current_s[idx] = c
        
        # Add contributions of new ABCs in the affected range
        new_contribution = sum([check_abc(current_s, i) for i in range(idx - 2, idx + 1)])
        
        new_count = current_count - old_contribution + new_contribution
        return (current_s, new_count)

    # Use reduce to iterate through queries and maintain state
    # We need to capture the count at each step, so we store the results in a list
    # Since reduce only returns the final state, we use a helper to store history.
    
    results = []
    def reduce_with_history(state, query):
        new_state = process_query(state, query)
        results.append(new_state[1])
        return new_state

    reduce(reduce_with_history, queries, (S, initial_count))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()