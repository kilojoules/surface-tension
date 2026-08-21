import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. 
    # Due to the symmetry of the set of all good sequences, 
    # if we replace every element x in a sequence with (N + 1 - x),
    # we get another good sequence.
    # This operation is a bijection that reverses the lexicographical order.
    # The sequence that is its own "complement" (or the one closest to it)
    # will be the middle one.
    # However, the simplest way to find the middle sequence is to 
    # construct it greedily. For each position, we try digits 1 to N.
    # We can skip a digit if the number of sequences starting with the 
    # current prefix is less than the remaining rank we need.
    
    # Since we need the floor((S+1)/2)-th, and S can be huge,
    # we use the property that the middle sequence is the one that 
    # "balances" the distribution of digits.
    # For a given prefix, the number of ways to complete it is 
    # (sum(remaining_counts))! / product(remaining_counts!)
    
    # To avoid huge numbers in the loop, we can use the fact that 
    # we are looking for the median.
    # The middle sequence is the one where we pick the smallest possible 
    # digit such that the number of sequences starting with that digit 
    # is at least the remaining rank.
    
    # Let's define a helper to calculate the number of permutations of a multiset.
    # Instead of calculating the full S, we can maintain the rank.
    # But S is too large for standard floats. We use Python's arbitrary precision integers.
    
    # Precompute factorials for the multiset permutation formula
    # Max NK is 250,000. We cannot precompute all factorials.
    # But we only need to compare the rank with the number of permutations.
    
    # Wait, there is a mathematical shortcut.
    # The middle sequence is the one that is "lexicographically" 
    # the average of the first and the last.
    # The first is (1*K, 2*K, ..., N*K)
    # The last is (N*K, (N-1)*K, ..., 1*K)
    # The middle sequence is the one that starts with the digit that 
    # splits the total count S into two halves.
    
    # Let's use the property: the middle sequence is the one where 
    # we pick digit 'd' such that the sum of permutations for digits < d 
    # is < S/2 and the sum for digits <= d is >= S/2.
    
    # Since we cannot use loops, we use reduce to iterate through the NK positions.
    # We need to keep track of: (current_rank, current_counts)
    # current_counts is a list of remaining counts for each digit 1..N.
    
    # To handle the large numbers, we use the formula:
    # Ways(counts) = (sum(counts))! / product(counts!)
    # We can compute this using a helper function.
    
    import math
    
    def get_ways(counts):
        total = sum(counts)
        # Using math.factorial is allowed as it's a builtin
        # But we need to avoid loops. We can use map/reduce for the product.
        denom = reduce(lambda a, b: a * math.factorial(b), counts, 1)
        return math.factorial(total) // denom

    # We need the total S to find the starting rank.
    # S = get_ways([k] * n)
    # target_rank = (S + 1) // 2
    
    # However, we can't use a loop to find the digit.
    # We can use a list comprehension to find the first digit that satisfies the condition.
    # But we need to do this for NK positions.
    
    # Let's refine the state for reduce: (rank, counts, result_sequence)
    # For each position i from 0 to NK-1:
    #   Find digit d in 1..N such that:
    #   sum(get_ways(counts with digit j at i) for j < d) < rank <= sum(...) for j <= d
    #   Update rank: rank = rank - sum(get_ways(...) for j < d)
    #   Update counts: counts[d-1] -= 1
    #   Append d to result_sequence
    
    # To avoid loops, we use a list comprehension to find the digit.
    # Since we can't use a loop to find the index, we can use a trick with 
    # a list of cumulative sums and then find the index using a generator expression.
    
    def step(state, _):
        rank, counts, seq = state
        # Calculate ways for each possible digit 1..N
        # ways_per_digit[j] is the number of sequences starting with digit j+1
        ways_per_digit = [
            get_ways(counts[:j] + [counts[j]-1] + counts[j+1:]) 
            if counts[j] > 0 else 0 
            for j in range(n)
        ]
        
        # Find the digit d (1-indexed)
        # We need the smallest d such that sum(ways_per_digit[:d]) >= rank
        # We can use a generator expression with next() to find the index
        d = next(j + 1 for j in range(n) if sum(ways_per_s := ways_per_digit[:j+1]) >= rank)
        
        # Update rank for the next position
        new_rank = rank - sum(ways_per_digit[:d-1])
        
        # Update counts
        new_counts = counts[:d-1] + [counts[d-1]-1] + counts[d:]
        
        return (new_rank, new_counts, seq + [d])

    # Initial state
    total_s = get_ways([k] * n)
    initial_rank = (total_s + 1) // 2
    initial_state = (initial_rank, [k] * n, [])
    
    # Use reduce to simulate the loop over NK positions
    final_state = reduce(step, range(n * k), initial_state)
    
    # Print the result sequence
    print(*(final_state[2]))

# Standard Python entry point
if __name__ == "__main__":
    solve()