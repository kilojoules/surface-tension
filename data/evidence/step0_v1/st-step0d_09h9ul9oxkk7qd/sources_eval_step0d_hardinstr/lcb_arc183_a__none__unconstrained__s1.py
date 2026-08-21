import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    
    # To avoid calculating S explicitly (which is huge), we can determine 
    # the sequence element by element.
    # For the first position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with v is ( (N*K - 1)! ) / ( (K-1)! * (K!)^(N-1) )
    # This is equal to S * (K / (N*K)) = S / N.
    
    # However, the target index is (S+1)//2.
    # If we use a 1-based index for the target:
    # For the first element v, the range of indices is:
    # v=1: [1, S/N]
    # v=2: [S/N + 1, 2S/N]
    # ...
    # v=N: [(N-1)S/N + 1, S]
    
    # The target index (S+1)//2 falls into the range of v = (N+1)//2 
    # if we consider the symmetry of the distribution.
    # Actually, the "middle" sequence of a symmetric set of permutations 
    # is the one that is its own "complement" (replacing i with N-i+1).
    # For any sequence Seq, its complement Seq' is also a good sequence.
    # Seq < Seq' if at the first index i where they differ, Seq_i < Seq'_i.
    # The (S+1)//2-th sequence is the one that is lexicographically 
    # just before or equal to its complement.
    
    # Specifically, for the middle sequence, at each step we want to pick 
    # the smallest available digit v such that the number of sequences 
    # starting with digits < v is less than (S+1)//2.
    
    # Let's use the property: the (S+1)//2-th sequence is the one where
    # we effectively "split the difference".
    # For the first digit, we check if (S+1)//2 <= S/N. 
    # If yes, digit is 1. If (S+1)//2 > S/N, we subtract S/N and move to digit 2.
    
    # Since we can't use loops, we use reduce to maintain the state:
    # state = (current_counts, target_index)
    # current_counts: list of remaining counts for each digit 1...N
    # target_index: the 1-based index of the sequence we are looking for
    
    def get_multiset_coeff(counts):
        # Total permutations of multiset: (sum(counts))! / product(counts!)
        # We can calculate this using combinations:
        # comb(n, k1) * comb(n-k1, k2) * ...
        return reduce(lambda acc, c: acc * comb(sum(counts) - (sum(counts) - sum(c for c in counts if c > 0)), c), 
                      [0], # This is a placeholder, logic needs to be inside reduce
                      1)

    # Correct way to calculate multiset coefficient without loops:
    def calc_s(counts):
        total = sum(counts)
        # Use a helper to calculate product of combs
        # We use a list of counts and reduce to multiply combinations
        return reduce(lambda a, c: a * comb(total - (sum(counts) - sum(counts)), c), counts, 1)
    
    # The above calc_s is wrong. Correct:
    def calc_s_fixed(counts):
        # Total = n! / (k1! k2! ... kn!)
        # = comb(n, k1) * comb(n-k1, k2) * ...
        # We can't use a loop, so we use reduce with a running total of remaining slots.
        def step(state, c):
            rem, res = state
            return (rem - c, res * comb(rem, c))
        return step(reduce(step, counts, (sum(counts), 1)), 0)[1]

    # To find the (S+1)//2-th sequence:
    # S = calc_s_fixed([K]*N)
    # target = (S + 1) // 2
    
    # We need to find the sequence. Since we can't use loops, 
    # we use reduce over the range of the total length NK.
    
    def find_sequence(N, K):
        total_s = calc_s_fixed([K]*N)
        target = (total_s + 1) // 2
        
        def solve_step(state, _):
            counts, t = state
            # Try digits v = 1 to N
            # We need to find v such that sum_{i=1}^{v-1} count_i * S_rem < t <= sum_{i=1}^{v} count_i * S_rem
            # where S_rem is the number of sequences possible with the remaining counts
            # after picking digit v.
            
            # Since we can't loop, we use another reduce to find the digit v
            def find_v(v_state, v):
                curr_t, chosen_v = v_state
                if chosen_v is not None:
                    return v_state
                
                # Number of sequences if we pick digit v
                # Remaining counts: counts[v-1]-1, others same
                # Ways = (Total-1)! / ( (counts[v-1]-1)! * product(others!) )
                # Ways = S_total_rem * (counts[v-1] / Total)
                
                # But it's easier to calculate S_rem directly:
                # If counts[v-1] == 0, ways = 0
                if counts[v-1] == 0:
                    return (curr_t, None)
                
                # Calculate S_rem for picking v
                # We create a temporary list for counts
                temp_counts = list(counts)
                temp_counts[v-1] -= 1
                ways = calc_s_fixed(temp_counts)
                
                if curr_t <= ways:
                    return (curr_t, v)
                else:
                    return (curr_t - ways, None)
            
            # Find v by reducing over range(1, N+1)
            final_v_state = reduce(find_v, range(1, N + 1), (t, None))
            v = final_v_state[1]
            
            # Update counts for the next step
            new_counts = list(counts)
            new_counts[v-1] -= 1
            # Update target t for the next step
            # The new t is the t used when v was finally picked
            # We need to track the t inside find_v.
            # Let's redefine find_v to return the updated t.
            
            # To get the updated t, we can't easily. Let's refine.
            return (new_counts, final_v_state[0]), v

        # We need to pass the target t through. 
        # The state in the outer reduce is (counts, t)
        # The result of the outer reduce is (final_state, sequence)
        
        def outer_step(state, _):
            (counts, t), seq = state
            
            def inner_step(v_state, v):
                curr_t, chosen_v = v_state
                if chosen_v is not None:
                    return (curr_t, chosen_v)
                if counts[v-1] == 0:
                    return (curr_t, None)
                
                temp_counts = list(counts)
                temp_counts[v-1] -= 1
                ways = calc_s_fixed(temp_counts)
                
                if curr_t <= ways:
                    return (curr_t, v)
                else:
                    return (curr_t - ways, None)
            
            res_v = reduce(inner_step, range(1, N + 1), (t, None))
            v = res_v[1]
            new_t = res_v[0]
            new_counts = list(counts)
            new_counts[v-1] -= 1
            return ((new_counts, new_t), seq + [v])

        initial_state = (([K]*N, target), [])
        final_result = reduce(outer_step, range(N * K), initial_state)
        return final_result[1]

    print(*(find_sequence(N, K)))

if __name__ == "__main__":
    solve()