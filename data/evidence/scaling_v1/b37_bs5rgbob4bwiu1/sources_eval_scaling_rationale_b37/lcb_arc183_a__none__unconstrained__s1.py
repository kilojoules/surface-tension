import sys
from functools import reduce
from math import factorial

# To solve this without loops or recursion, we use reduce to simulate the 
# process of filling the sequence position by position.
# The total number of good sequences S is (N*K)! / (K!)^N.
# We need the floor((S+1)/2)-th sequence.

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # Precompute factorials for multiset permutation formula:
    # Count = (sum of remaining)! / product(remaining_i!)
    # Since we only need to compare the target index with the count of sequences
    # starting with a certain number, we can calculate the count dynamically.
    
    # Initial state for reduce: 
    # (current_target_index, current_counts_of_numbers, result_sequence)
    # target_index is 1-based.
    
    # Total sequences S = (n*k)! / (k!**n)
    # Target index = (S + 1) // 2
    
    # We use a helper to calculate permutations of a multiset
    def count_permutations(counts):
        total = sum(counts)
        # Formula: total! / (c1! * c2! ... * cn!)
        # We use a comprehension and math.prod (available in Python 3.8+)
        import math
        denom = reduce(lambda x, y: x * math.factorial(y), counts, 1)
        return math.factorial(total) // denom

    # Calculate initial S
    import math
    s_total = math.factorial(n * k) // (math.factorial(k)**n)
    target = (s_total + 1) // 2

    # We need to determine the character for each of the N*K positions.
    # range(n * k) provides the sequence of positions.
    # The accumulator carries (target, counts, sequence)
    
    # To avoid loops, we use reduce over the range of total length.
    # Inside reduce, we need to find the smallest i such that 
    # sum of permutations for 1..i >= target.
    
    # Since we cannot use loops, we use a list comprehension to calculate 
    # the number of permutations for each possible next digit (1 to N).
    
    def get_next_digit(state):
        target_idx, counts, seq = state
        
        # Calculate how many sequences start with digit d (1 to N)
        # if counts[d-1] > 0, then count is count_permutations(counts with d decremented)
        
        # Create a list of (digit, count_of_sequences_starting_with_digit)
        # We use a list comprehension here.
        options = [
            (d, count_permutations([counts[i] - (1 if i == d-1 else 0) for i in range(n)]))
            if counts[d-1] > 0 else (d, 0)
            for d in range(1, n + 1)
        ]
        
        # Find the digit d where the cumulative sum of options reaches target_idx
        # We use a list comprehension to create cumulative sums
        cum_sums = reduce(lambda acc, x: acc + [acc[-1] + x[1]], options, [0])
        
        # Find the first index where cum_sums[i+1] >= target_idx
        # We use next() with a generator expression to find the digit
        digit = next(options[i][0] for i in range(n) if cum_sums[i+1] >= target_idx)
        
        # Update target_idx: subtract the sum of sequences for digits < chosen digit
        new_target = target_idx - cum_sums[range(n)[0] if n==0 else [i for i in range(n) if options[i][0] == digit][0]]
        # Wait, the above logic for new_target is messy. Let's refine:
        # The sum of sequences for digits < chosen digit is cum_sums[index_of_digit]
        idx_of_digit = [i for i in range(n) if options[i][0] == digit][0]
        new_target = target_idx - cum_sums[idx_of_digit]
        
        new_counts = [counts[i] - (1 if i == digit-1 else 0) for i in range(n)]
        
        return (new_target, new_counts, seq + [digit])

    # Final execution using reduce
    final_state = reduce(lambda state, _: get_next_digit(state), range(n * k), (target, [k] * n, []))
    
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()