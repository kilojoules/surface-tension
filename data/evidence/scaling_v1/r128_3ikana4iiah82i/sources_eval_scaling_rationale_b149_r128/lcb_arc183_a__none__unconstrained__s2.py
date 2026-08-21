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

    # Total number of good sequences is (N*K)! / (K!)^N
    # We need the floor((S + 1) / 2)-th sequence.
    # S = total permutations of the multiset {1*K, 2*K, ..., N*K}
    
    # To find the rank, we first calculate the total number of sequences S.
    # However, S can be massive, so we work with the target rank directly.
    # The total number of sequences is S. We want the sequence at rank (S + 1) // 2.
    
    # We can determine the sequence greedily. For the first position, we try 
    # numbers 1 to N. If we pick number 'i', the number of ways to complete 
    # the sequence is (Remaining_Total)! / Product(Remaining_Counts!)
    
    # Instead of calculating S first, we can observe that the "middle" sequence
    # is reached when we have exhausted half of the total permutations.
    
    # Let's use a helper to calculate permutations of a multiset.
    def count_permutations(counts):
        total = sum(counts)
        # Using a property: Permutations = total! / (c1! * c2! ... cn!)
        # We can calculate this using combinations: comb(total, c1) * comb(total-c1, c2) ...
        return reduce(lambda acc, c: acc * comb(total - (sum(counts) - total), c), counts, 1)
    
    # Wait, the above reduce logic for count_permutations is slightly flawed. 
    # Correct way:
    def get_total_perms(counts):
        t = sum(counts)
        # Permutations = comb(t, counts[0]) * comb(t-counts[0], counts[1]) ...
        # We use a running total for the combinations.
        def folder(acc, c):
            # acc is (current_total, current_product)
            return (acc[0] - c, acc[1] * comb(acc[0], c))
        
        return folder(reduce(folder, counts, (t, 1)))[1]

    # The target rank is (S + 1) // 2.
    # Since we need the middle, we can track the "remaining rank" needed.
    # Initial rank needed: (get_total_perms([k]*n) + 1) // 2
    
    # To avoid calculating S explicitly at the start (which is huge), 
    # we can use the fact that we want the middle.
    # For the first digit, we check if the number of sequences starting with 1, 2... 
    # is less than the target rank.
    
    # Let's pre-calculate the target rank.
    total_s = get_total_perms([k] * n)
    target_rank = (total_s + 1) // 2
    
    # We use reduce to build the sequence.
    # State: (current_counts, current_rank, result_sequence)
    def build_sequence(state, _):
        counts, rank, seq = state
        # Try numbers i from 1 to N
        def find_digit(digit, current_rank):
            if counts[digit-1] == 0:
                return (digit, current_rank, False)
            
            # Number of sequences if we pick 'digit'
            # Remaining counts: counts with counts[digit-1] decremented
            rem_counts = list(counts)
            rem_counts[digit-1] -= 1
            num_ways = get_total_perms(rem_counts)
            
            if current_rank <= num_ways:
                return (digit, current_rank, True)
            else:
                return (digit, current_rank - num_ways, False)

        # We need to find the first digit i that satisfies the rank.
        # We use a small loop-like structure via reduce to find the digit.
        def search_digit(state_search, i):
            # state_search: (found, digit, updated_rank)
            if state_search[0]: return state_search
            res_digit, res_rank, found = find_digit(i, state_search[2])
            return (found, res_digit, res_rank) if found else (False, 0, res_rank)
        
        search_res = reduce(search_digit, range(1, n + 1), (False, 0, rank))
        
        # Update state for next position
        final_digit = search_res[1]
        new_counts = list(counts)
        new_counts[final_digit-1] -= 1
        return (new_counts, search_res[2], seq + [final_digit])

    # The sequence length is N*K
    final_state = reduce(build_sequence, range(n * k), ([k] * n, target_rank, []))
    
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()