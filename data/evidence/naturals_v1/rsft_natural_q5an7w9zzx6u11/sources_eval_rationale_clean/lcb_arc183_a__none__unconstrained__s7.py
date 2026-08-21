import sys
from functools import reduce
from math import factorial

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    # However, S can be massive, so we handle the target index carefully.
    
    # Precompute factorials for the multinomial coefficient
    # The number of ways to arrange items with counts c1, c2, ..., cN is
    # (sum(ci))! / (c1! * c2! * ... * cN!)
    
    # Since we need the "middle" sequence, we can observe a symmetry:
    # If a sequence is (s1, s2, ..., sn), its "complement" is (N+1-s1, N+1-s2, ..., N+1-sn).
    # The lexicographical middle is related to this symmetry.
    # Specifically, the S-th sequence is the reverse-complement of the 1st.
    # The floor((S+1)/2)-th sequence is the one that, when paired with its 
    # complement, is the smaller of the two (or the middle one).
    # This means for every position i, we want to pick the smallest digit d 
    # such that the number of sequences starting with digits < d is less than 
    # half of the total remaining sequences.
    
    # Actually, a simpler observation: the floor((S+1)/2)-th sequence is 
    # the one where we effectively try to pick the "median" digit at each step.
    # Because of the symmetry of the set of all good sequences, the 
    # floor((S+1)/2)-th sequence is the one that is lexicographically 
    # just before or equal to its complement (N+1-s_i).
    
    # For a sequence to be <= its complement, at the first index i where they differ,
    # s_i < N+1-s_i.
    # This is equivalent to saying we want the largest sequence that is 
    # lexicographically smaller than or equal to its complement.
    
    # The most direct way to find the floor((S+1)/2)-th sequence is to 
    # realize that the set of all good sequences is symmetric.
    # The sequence at index (S+1)//2 is the one that "balances" the distribution.
    # This is achieved by picking the digit d at each step such that the 
    # number of sequences starting with digits < d is < S/2, 
    # and the number of sequences starting with digits <= d is >= S/2.
    
    # Given the constraints and the "no loop" rule, we use reduce to maintain 
    # (current_counts, current_target_index).
    
    def get_count(counts):
        # Multinomial coefficient: (sum(counts))! / product(c!)
        total = sum(counts)
        res = factorial(total)
        # Use a list comprehension and reduce to calculate product of factorials
        denom = reduce(lambda x, y: x * y, [factorial(c) for c in counts], 1)
        return res // denom

    # Initial state: counts of each number 1..N, and the target index
    # Total S = get_count([K]*N)
    # Target = (S + 1) // 2
    
    initial_counts = tuple([K] * N)
    total_S = get_count(list(initial_counts))
    target_idx = (total_S + 1) // 2
    
    def step(state, _):
        counts, target = state
        # We need to find the smallest digit d (1-indexed) such that
        # sum_{i=1}^{d-1} count(i) * get_count(counts - e_i) < target
        # and sum_{i=1}^{d} count(i) * get_count(counts - e_i) >= target
        
        # Calculate the number of permutations for each possible next digit
        # ways[d] = number of sequences starting with digit d+1
        ways = [
            (counts[d] * get_count(list(counts[:d] + (counts[d]-1,) + counts[d+1:])) 
             if counts[d] > 0 else 0) 
            for d in range(N)
        ]
        
        # Find the digit d using a list comprehension to find the first index 
        # where the prefix sum of ways reaches target
        # We use a helper to get prefix sums without loops
        prefix_sums = list(reduce(lambda acc, x: acc + [acc[-1] + x], ways, [0]))
        
        # The digit d is the first index where prefix_sums[d+1] >= target
        # Since we can't use loops, we use a list comprehension to find the index
        d = [i for i in range(N) if prefix_sums[i] < target <= prefix_sums[i+1]][0]
        
        # Update counts and target for the next position
        new_counts = list(counts)
        new_counts[d] -= 1
        new_target = target - prefix_sums[d]
        
        return (tuple(new_counts), new_target), d + 1

    # Use reduce to iterate NK times
    # The result of reduce is ((final_counts, final_target), last_digit)
    # But we need all digits, so we use a trick with a list and reduce
    
    def solve_recursive(state, _):
        (counts, target), result = state
        # Calculate ways for the current state
        ways = [
            (counts[d] * get_count(list(counts[:d] + (counts[d]-1,) + counts[d+1:])) 
             if counts[d] > 0 else 0) 
            for d in range(N)
        ]
        prefix_sums = list(reduce(lambda acc, x: acc + [acc[-1] + x], ways, [0]))
        d = [i for i in range(N) if prefix_sums[i] < target <= prefix_sums[i+1]][0]
        
        new_counts = list(counts)
        new_counts[d] -= 1
        new_target = target - prefix_sums[d]
        
        return ((tuple(new_counts), new_target), result + [d + 1])

    final_state = reduce(solve_recursive, range(N * K), ((initial_counts, target_idx), []))
    print(*(final_state[1]))

if __name__ == "__main__":
    solve()