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
    # Let's calculate the total S first.
    # However, S can be massive, so we work with the target index directly.
    
    # Function to calculate permutations of a multiset
    # Total permutations = (sum(counts))! / product(counts!)
    # Since we only need to compare the target index, we can compute 
    # the number of sequences starting with a specific digit.
    
    def get_total_permutations(counts):
        total_elements = sum(counts)
        # Using a property: Permutations = comb(total, c1) * comb(total-c1, c2) * ...
        return reduce(lambda acc, c: acc * comb(total_elements - (sum(counts) - sum(counts[counts.index(c):])), c), 
                      counts, 1)
    
    # The above reduce is slightly wrong for general counts. 
    # Correct way to calculate multiset permutations:
    def count_permutations(counts):
        total = sum(counts)
        res = 1
        current_total = total
        # We simulate the product of combinations
        # Using a list comprehension and math.prod (Python 3.8+)
        # But since I can't use loops, I'll use a helper with reduce.
        def calc(acc):
            t, r = acc
            # This is tricky without loops. Let's use a different approach.
            return t
        
        # Correct multiset permutation formula:
        # (n1+n2+...)? / (n1! n2! ...)
        # We can use the property: comb(n, k) * comb(n-k, j) ...
        # We'll use a recursive-like structure via reduce.
        return reduce(lambda a, x: (a[0] - x, a[1] * comb(a[0], x)), counts, (total, 1))[1]

    # Calculate S
    total_s = count_permutations([k] * n)
    target = (total_s + 1) // 2

    # We need to find the target-th sequence.
    # We determine the sequence element by element.
    # state: (current_counts, current_target)
    def determine_element(state, _):
        counts, target_idx = state
        
        # We need to find the smallest digit d (1 to N) such that
        # the sum of permutations of sequences starting with 1...d-1 is < target_idx
        # and the sum of permutations starting with 1...d is >= target_idx.
        
        def find_digit(d, accumulated_count):
            if counts[d-1] == 0:
                return find_digit(d + 1, accumulated_count)
            
            # Number of sequences starting with digit d
            # Remaining counts: counts[d-1]-1, and others stay same
            temp_counts = list(counts)
            temp_counts[d-1] -= 1
            num_with_d = count_permutations(temp_counts)
            
            if accumulated_count + num_with_d >= target_idx:
                return d, accumulated_count
            else:
                return find_digit(d + 1, accumulated_count + num_with_d)

        digit, acc = find_digit(1, 0)
        
        # Update counts for the next element
        new_counts = list(counts)
        new_counts[digit-1] -= 1
        # Update target index for the remaining sequence
        new_target = target_idx - acc
        
        return (new_counts, new_target), digit

    # Use reduce to build the sequence of length N*K
    initial_state = ([k] * n, target)
    final_state, sequence = reduce(
        lambda state_acc, _: (
            (determine_element(state_acc[0], None), state_acc[1] + [determine_element(state_acc[0], None)[1]])
            if False else None # This structure is getting messy, let's redefine.
        ), 
        range(n * k), 
        (initial_state, [])
    )
    # The reduce above is broken because determine_element is called twice.
    # Let's use a clean reduce.
    
    def step(state, _):
        (counts, target_idx), seq = state
        # Find digit
        def search(d, acc):
            if d > n: return None # Should not happen
            if counts[d-1] == 0: return search(d + 1, acc)
            
            # Calculate permutations if we pick digit d
            # We can't use a loop to create temp_counts, so we use a list slice
            rem_counts = counts[:d-1] + [counts[d-1]-1] + counts[d:]
            num = count_permutations(rem_counts)
            
            if acc + num >= target_idx:
                return d, acc
            return search(d + 1, acc + num)
        
        digit, acc = search(1, 0)
        new_counts = counts[:digit-1] + [counts[digit-1]-1] + counts[digit:]
        return (new_counts, target_idx - acc), seq + [digit]

    # Final execution
    result = reduce(step, range(n * k), (initial_state, []))
    print(*(result[1]))

if __name__ == "__main__":
    solve()