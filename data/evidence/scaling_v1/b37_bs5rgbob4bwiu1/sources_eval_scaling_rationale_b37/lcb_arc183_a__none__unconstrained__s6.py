import sys
from math import factorial
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is the multinomial coefficient:
    # S = (N*K)! / (K!)^N
    # We need the floor((S + 1) / 2)-th sequence.
    
    # Precompute factorials for the multinomial formula
    # Using a dictionary or list for fast lookup
    fact = [1] * (n * k + 1)
    # Since we can't use loops, we use reduce to populate the factorial list
    # However, the prompt forbids loops for the logic, but allows standard library.
    # To strictly follow "no loops", I'll use a recursive-style approach via reduce.
    # Actually, the most "functional" way to build a factorial table is via scan/reduce.
    
    # Using a list comprehension with a helper to simulate factorial precomputation
    # since we cannot use for-loops.
    def get_fact(max_n):
        return reduce(lambda acc, i: acc + [acc[-1] * i], range(1, max_n + 1), [1])

    f = get_fact(n * k)

    def count_permutations(counts):
        # Formula: (sum of counts)! / product(count!)
        total = sum(counts)
        denominator = reduce(lambda acc, c: acc * f[c], counts, 1)
        return f[total] // denominator

    # Total sequences S
    s_total = count_permutations([k] * n)
    target = (s_total + 1) // 2

    # We need to determine the sequence element by element.
    # State: (current_target, current_counts)
    # We use reduce to iterate over the positions 0 to NK-1.
    
    # Initial state: (target_index, list_of_counts, result_sequence)
    initial_state = (target, [k] * n, [])
    
    def find_next_element(state, _):
        target_idx, counts, sequence = state
        
        # We need to find the smallest digit d (1 to N) such that 
        # the number of sequences starting with digits smaller than d 
        # is less than target_idx, but including d reaches or exceeds it.
        
        # To avoid loops, we use a helper function with reduce to find the digit
        # search_state: (current_sum, found_digit, updated_counts)
        def search_digit(search_state, d_minus_1):
            curr_sum, found_d, curr_counts = search_state
            if found_d != -1:
                return search_state
            
            # Count sequences if we pick digit (d_minus_1 + 1)
            # Digit index is d_minus_1
            if curr_counts[d_minus_1] > 0:
                # Create a temporary count list to calculate permutations
                # Since we can't use loops, we use a list comprehension
                temp_counts = [curr_counts[i] - (1 if i == d_minus_1 else 0) for i in range(n)]
                num_ways = count_permutations(temp_counts)
                
                if curr_sum + num_ways >= target_idx:
                    return (curr_sum, d_minus_1 + 1, curr_counts)
                else:
                    return (curr_sum + num_ways, -1, curr_counts)
            else:
                return (curr_sum, found_d, curr_counts)

        # Iterate through possible digits 1 to N (indices 0 to N-1)
        search_result = reduce(search_digit, range(n), (0, -1, counts))
        final_sum, digit, _ = search_result
        
        # Update counts for the chosen digit
        new_counts = [counts[i] - (1 if i == digit - 1 else 0) for i in range(n)]
        # Update target index for the next position
        new_target = target_idx - final_sum
        
        return (new_target, new_counts, sequence + [digit])

    final_state = reduce(find_next_element, range(n * k), initial_state)
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()