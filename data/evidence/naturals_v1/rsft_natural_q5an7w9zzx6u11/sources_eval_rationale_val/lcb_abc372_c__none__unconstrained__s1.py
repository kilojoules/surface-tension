import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    N = int(input_data[0])
    Q = int(input_data[1])
    S_init = list(input_data[2])
    
    # Queries are pairs of (index, char) starting from index 3 of input_data
    queries = [
        (int(input_data[i]), input_data[i+1]) 
        for i in range(3, len(input_data), 2)
    ]

    # Helper to count "ABC" occurrences in a small window around index i
    # We check windows of size 3 that could overlap with index i
    # Indices to check as starts of "ABC": i-2, i-1, i
    def count_abc(s, i):
        return sum(
            1 for start in (i-2, i-1, i)
            if 0 <= start <= N - 3 and 
            s[start] == 'A' and s[start+1] == 'B' and s[start+2] == 'C'
        )

    # Initial count of "ABC"
    initial_count = sum(
        1 for i in range(N - 2) 
        if S_init[i] == 'A' and S_init[i+1] == 'B' and S_init[i+2] == 'C'
    )

    # Use reduce to process queries and maintain (current_S, current_count)
    # We use a list for S to allow mutation, but we wrap the logic in a function
    def process_query(state, query):
        S, count = state
        idx, char = query
        pos = idx - 1 # 1-indexed to 0-indexed
        
        # 1. Subtract occurrences that will be destroyed by the change
        count -= count_abc(S, pos)
        
        # 2. Update the character
        S[pos] = char
        
        # 3. Add occurrences created by the change
        count += count_abc(S, pos)
        
        # We print the count immediately; since we can't use a loop, 
        # we use a list comprehension or map to trigger the print
        [sys.stdout.write(f"{count}\n")]
        
        return (S, count)

    # Start the reduction
    reduce(process_query, queries, (S_init, initial_count))

if __name__ == "__main__":
    solve()