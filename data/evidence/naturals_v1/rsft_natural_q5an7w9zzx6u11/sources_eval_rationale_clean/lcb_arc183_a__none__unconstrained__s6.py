import sys
from functools import reduce
from math import factorial

# Increase recursion depth for deep reduce/map calls if necessary
sys.setrecursionlimit(2000)

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # Function to calculate the number of good sequences given remaining counts
    # Formula: (sum of counts)! / (product of (count!))
    def count_permutations(counts):
        total = sum(counts)
        denom = reduce(lambda x, y: x * factorial(y), counts, 1)
        return factorial(total) // denom

    # The total number of good sequences S
    # S = (N*K)! / (K!^N)
    total_S = count_permutations([K] * N)
    
    # We need the floor((S+1)/2)-th sequence.
    # In 0-indexing, this is floor((S+1)/2) - 1.
    target_idx = (total_S + 1) // 2 - 1

    # We use reduce to build the sequence one by one.
    # State: (current_counts, current_target_idx, result_sequence)
    # current_counts: list of remaining occurrences for each number 1...N
    def find_next_digit(state, _):
        counts, target, seq = state
        
        # We need to find which digit d (1 to N) fits at the current position.
        # For each digit d, if we place it, there are count_permutations(updated_counts) 
        # sequences starting with the current prefix.
        
        def search_digit(d, current_target):
            if d > N:
                return None # Should not happen
            
            if counts[d-1] > 0:
                # Calculate permutations if we pick digit d
                # Temporarily decrement count for d
                temp_counts = list(counts)
                temp_counts[d-1] -= 1
                num_ways = count_permutations(temp_counts)
                
                if current_target < num_ways:
                    return (d, current_target)
                else:
                    # Try next digit, subtracting the ways covered by digit d
                    return search_digit(d + 1, current_target - num_ways)
            else:
                # Digit d is exhausted, try next
                return search_digit(d + 1, current_target)

        digit, new_target = search_digit(1, target)
        
        # Update counts for the next step
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        
        return (new_counts, new_target, seq + [digit])

    # Initial state: counts of each digit, target index, empty sequence
    initial_state = ([K] * N, target_idx, [])
    
    # We need to run the reduce NK times. 
    # We use range(N * K) as the iterable to drive the process.
    final_state = reduce(find_next_digit, range(N * K), initial_state)
    
    # The result sequence is the third element of the final state tuple
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()