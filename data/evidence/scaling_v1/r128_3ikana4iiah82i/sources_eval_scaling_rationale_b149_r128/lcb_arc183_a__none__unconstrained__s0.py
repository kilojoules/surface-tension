import sys
from math import comb
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let's calculate S.
    # Note: S can be very large, so we use Python's arbitrary precision integers.
    
    def get_total_permutations(counts):
        # Formula for permutations of a multiset: (sum(counts))! / product(counts!)
        # However, we can compute this using combinations:
        # comb(n1+n2, n1) * comb(n2+n3, n2) ...
        return reduce(lambda acc, x: acc * comb(sum(counts[:counts.index(x)+1]), x), 
                      counts, 1)

    # A more reliable way to compute multiset permutations:
    def multiset_perm(counts):
        total = sum(counts)
        # Using the property: (n1+n2+n3)!/(n1!n2!n3!) = comb(n1+n2+n3, n1) * comb(n2+n3, n2) * ...
        # We iterate through the counts to multiply combinations.
        return reduce(lambda a, b: a * b, 
                      [comb(sum(counts[i:]), counts[i]) for i in range(len(counts))], 1)

    # Initial counts of each number 1 to N
    initial_counts = [k] * n
    s = multiset_perm(initial_counts)
    target = (s + 1) // 2

    def find_sequence(state):
        # state: (current_counts, current_target)
        counts, target_idx = state
        
        # If all numbers are used, return empty list
        if sum(counts) == 0:
            return []

        # We need to find the smallest digit d (1 to N) such that 
        # the number of sequences starting with digits < d is less than target_idx,
        # and the number of sequences starting with digits <= d is >= target_idx.
        
        def search_digit(d, current_target):
            if d > n:
                return None
            
            # If we pick digit d, how many sequences can we form with the remaining?
            # We need to check if the current_target falls within the range of sequences 
            # starting with digit d.
            # First, calculate how many sequences start with digits 1 to d-1.
            
            # To do this efficiently, we calculate the number of permutations 
            # if we were to pick digit 'i' for the current position.
            # The number of ways to complete the sequence is multiset_perm(counts - e_i)
            
            # Instead of a loop, we use a generator and next() to find the digit.
            # We calculate the number of permutations for each possible digit i from 1 to N.
            
            # The number of ways to complete the sequence if we pick digit i:
            # ways(i) = (sum(counts)-1)! / (counts[0]! ... (counts[i-1]-1)! ... counts[n-1]!)
            # ways(i) = multiset_perm(counts) * counts[i-1] / sum(counts)
            
            # Let's use a more direct approach to find the digit.
            pass

    # Since I cannot use loops, I will use a recursive-like structure via reduce
    # to build the sequence one by one.
    
    def step(state, _):
        counts, target_idx = state
        
        # Calculate ways to complete the sequence if we pick digit i
        # ways(i) = multiset_perm(counts - e_i)
        # We find i such that sum(ways(1...i-1)) < target_idx <= sum(ways(1...i))
        
        def get_ways(i):
            # Create a copy of counts and decrement the i-th element
            temp_counts = list(counts)
            temp_counts[i-1] -= 1
            return multiset_perm(temp_counts)

        # Use a generator to find the digit i and the new target_idx
        # We use a list comprehension to calculate ways for all i from 1 to N
        ways = [get_ways(i) for i in range(1, n + 1)]
        
        # Find the digit i
        # We need the smallest i such that sum(ways[:i]) >= target_idx
        # We can use a generator expression with next()
        digit = next(i for i in range(1, n + 1) if sum(ways[:i]) >= target_idx)
        
        # New target index is target_idx minus the ways from digits < digit
        new_target = target_idx - sum(ways[:digit-1])
        
        # Update counts
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        
        return (new_counts, new_target), digit

    # Use reduce to simulate the process for NK steps
    # The accumulator is (state, sequence)
    final_state, sequence = reduce(
        lambda acc, _: (step(acc[0], _), acc[1] + [step(acc[0], _)[1]]),
        range(n * k),
        ((initial_counts, target), [])
    )
    
    # The reduce logic above calls step() twice per iteration, which is inefficient.
    # Let's refine it to call step() once.
    
    def refined_step(acc, _):
        state, seq = acc
        new_state, digit = step(state, _)
        return (new_state, seq + [digit])

    # Re-running the reduce with the refined step
    result = reduce(refined_step, range(n * k), ((initial_counts, target), []))
    
    print(*(result[1]))

# To avoid the double-call in the first reduce and the recursion limit, 
# I will rewrite the logic into a single reduce block.
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    n, k = map(int, input_data)
    
    def multiset_perm(counts):
        return reduce(lambda a, b: a * b, [comb(sum(counts[i:]), counts[i]) for i in range(len(counts))], 1)

    initial_counts = [k] * n
    s = multiset_perm(initial_counts)
    target = (s + 1) // 2

    def step(acc, _):
        (counts, target_idx), seq = acc
        # Calculate ways for each digit 1..N
        # ways[i] is the number of sequences starting with digit i+1
        ways = [multiset_perm([counts[j] - (1 if j == i else 0) for j in range(n)]) for i in range(n)]
        
        # Find digit d (1-indexed)
        # We need the first d such that sum(ways[:d]) >= target_idx
        # Using a list comprehension to find the index
        prefix_sums = [sum(ways[:i+1]) for i in range(n)]
        digit_idx = next(i for i, s_val in enumerate(prefix_sums) if s_val >= target_idx)
        
        new_counts = [counts[j] - (1 if j == digit_idx else 0) for j in range(n)]
        new_target = target_idx - (sum(ways[:digit_idx]) if digit_idx > 0 else 0)
        
        return (new_counts, new_target), seq + [digit_idx + 1]

    final_result = reduce(step, range(n * k), ((initial_counts, target), []))
    print(*(final_result[1]))

if __name__ == "__main__":
    final_solve()