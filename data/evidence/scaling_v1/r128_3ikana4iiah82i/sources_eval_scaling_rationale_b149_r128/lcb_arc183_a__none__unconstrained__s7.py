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
    # Let target = (S + 1) // 2.
    
    # To find the sequence, we determine each position one by one.
    # For a position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with the current prefix and v is:
    # (Remaining Length)! / (Product of factorials of remaining counts)
    
    # Instead of calculating large factorials, we can use the property:
    # Count(v) = [ (Total Remaining - 1)! / (Product of counts!) ] * (count of v)
    # Which is: Total_Permutations * (count of v) / (Total Remaining)
    
    # We use reduce to maintain a state: (current_counts, target_index, result_sequence)
    # current_counts: list of remaining counts for each number 1...N
    # target_index: the 1-based index of the sequence we are looking for
    
    initial_counts = [k] * n
    
    # Calculate total S to find the starting target index
    # S = (N*K)! / (K!)^N
    # However, we can calculate the target index for the first element directly.
    # The number of sequences starting with '1' is S * K / (N*K) = S / N.
    
    # Let's calculate S first.
    # S = comb(N*K, K) * comb((N-1)*K, K) * ... * comb(K, K)
    total_s = reduce(lambda a, b: a * comb(b, k), range(n*k, 0, -k), 1)
    target = (total_s + 1) // 2

    def get_next_state(state, _):
        counts, target_idx, res = state
        total_rem = sum(counts)
        
        # Calculate total permutations of the remaining multiset
        # Total = (sum counts)! / product(counts!)
        # We can compute this using a product of combinations
        curr_total_perm = reduce(lambda a, b: a * comb(b, counts[a]), range(total_rem, 0, -1), 1)
        # Wait, the above reduce logic for total_perm is slightly wrong. 
        # Correct way to calc multiset permutation:
        # total_perm = (sum(counts))! / product(c!)
        # But we only need to know if target_idx <= permutations starting with v.
        
        # Let's redefine: for v in 1..N, 
        # perms_starting_with_v = (total_rem - 1)! / (counts[v]-1)! * product(counts[i]! for i != v)
        # = [ (total_rem)! / product(counts[i]!) ] * counts[v] / total_rem
        
        # To avoid re-calculating the huge total_perm every time, 
        # we can pass it in the state.
        return state

    # Since the 'no loop' constraint is strict, I will use a recursive-like 
    # structure via reduce, but I need to calculate the permutations carefully.
    
    def find_sequence(state, _):
        counts, target_idx, res, total_perm = state
        total_rem = sum(counts)
        
        # We need to find v such that sum(perms(1...v-1)) < target_idx <= sum(perms(1...v))
        # perms(v) = total_perm * counts[v-1] // total_rem
        
        # We use a nested reduce to find the value v and the new target_idx
        # inner_state: (v, current_target, found)
        def find_v(inner_state, v):
            v_idx, curr_t, found = inner_state
            if found: return inner_state
            
            # count of v is counts[v-1]
            p_v = (total_perm * counts[v-1]) // total_rem
            if curr_t <= p_v:
                return (v, curr_t, True)
            else:
                return (v_idx, curr_t - p_v, False)

        # Use range(1, n+1) to find the character
        final_v_state = reduce(find_v, range(1, n + 1), (0, target_idx, False))
        v, new_target, _ = final_v_state
        
        # Update counts and total_perm for the next position
        # New total_perm = (total_rem - 1)! / (counts[v-1]-1)! * product(counts[i]! for i != v)
        # = total_perm * counts[v-1] // total_rem
        new_total_perm = (total_perm * counts[v-1]) // total_rem
        
        # Update counts list
        new_counts = list(counts)
        new_counts[v-1] -= 1
        
        return (new_counts, new_target, res + [v], new_total_perm)

    # Start reduce
    final_state = reduce(find_sequence, range(n * k), (initial_counts, target, [], total_s))
    
    # Print the result sequence
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()