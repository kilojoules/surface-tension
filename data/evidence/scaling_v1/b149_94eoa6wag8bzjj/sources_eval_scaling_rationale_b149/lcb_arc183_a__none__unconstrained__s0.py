import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Since we cannot use loops, we use reduce to build the sequence.
    # state: (current_counts, target_index)
    # current_counts: list of remaining counts for each number 1...N
    
    def get_total_permutations(counts):
        # Formula for permutations of a multiset: (sum(counts))! / product(counts[i]!)
        # However, we can compute this incrementally.
        # For the first position, if we pick number i, the remaining permutations are:
        # (Total-1)! / (K!^(N-1) * (K-1)!)
        # Which is (Total! / (K!^N)) * (K / Total)
        pass

    # To avoid heavy factorial math, we observe that the number of sequences 
    # starting with number 'i' is:
    # Total_Permutations * (remaining_count[i] / total_remaining_elements)
    
    # We need the index target = (S + 1) // 2
    # S = (N*K)! / (K!**N)
    # Instead of computing S, we can work with the target index directly.
    # But S can be massive, so we use Python's arbitrary precision integers.
    
    def factorial(n):
        return reduce(lambda x, y: x * y, range(1, n + 1), 1)

    S = factorial(N * K) // (factorial(K) ** N)
    target = (S + 1) // 2

    def get_count_with_prefix(counts):
        # Total permutations given remaining counts
        total_rem = sum(counts)
        if total_rem == 0:
            return 1
        # Using a more efficient way to calculate multiset permutations
        # Perm = (sum(c_i))! / product(c_i!)
        num = factorial(total_rem)
        den = reduce(lambda x, y: x * factorial(y), counts, 1)
        return num // den

    def step(state, _):
        counts, target_idx = state
        total_rem = sum(counts)
        
        # Find the smallest i such that the sum of permutations of sequences 
        # starting with 1...i is >= target_idx
        def find_digit(i, current_target):
            if i > N:
                return None, None
            if counts[i-1] == 0:
                return find_digit(i + 1, current_target)
            
            # Number of sequences starting with digit i:
            # (total_rem - 1)! / (counts[0]! ... (counts[i-1]-1)! ... counts[N-1]!)
            # = get_count_with_prefix(counts) * counts[i-1] // total_rem
            
            # To avoid recomputing get_count_with_prefix, we pass it or compute it once
            # But we are in a helper function.
            return i, 0 # Placeholder

        # Since we can't loop, we use a list comprehension or map to find the digit
        # Calculate permutations for each possible digit 1...N
        perms_for_digit = [
            (get_count_with_prefix(
                [counts[j] - (1 if j == i-1 else 0) for j in range(N)]) 
             if counts[i-1] > 0 else 0)
            for i in range(1, N + 1)
        ]
        
        # Find the digit by accumulating the counts
        # We use a trick with reduce to find the index where the target falls
        def find_idx(acc):
            # acc: (current_sum, chosen_digit)
            # We want the first digit where current_sum + perms >= target_idx
            # Because we can't loop, we use a list comprehension to find the first index
            return None

        # Correct way to find the digit without loops:
        # Create a list of cumulative sums
        cum_sums = list(reduce(lambda x, y: x + [x[-1] + y] if x else [y], perms_for_digit, []))
        # The digit is the first index where cum_sum >= target_idx
        digit = next(i + 1 for i, s in enumerate(cum_sums) if s >= target_idx)
        
        # Update target_idx for the next position
        prev_cum_sum = cum_sums[digit-2] if digit > 1 else 0
        new_target = target_idx - prev_cum_sum
        
        # Update counts
        new_counts = [counts[j] - (1 if j == digit-1 else 0) for j in range(N)]
        
        return (new_counts, new_target), digit

    # Use reduce to generate the sequence
    initial_state = ([K] * N, target)
    final_state, sequence = reduce(
        lambda acc, _: (step(acc[0], _), acc[1] + [step(acc[0], _)[1]]), 
        range(N * K), 
        (initial_state, [])
    )
    
    # The reduce above is slightly wrong because step is called twice. 
    # Let's fix it using a more robust reduce pattern.
    
    def solve_fixed():
        # Re-implementing the reduce to avoid double-calling step
        def folder(state, _):
            counts, target_idx = state[0]
            # Calculate perms for each digit
            perms = [
                (get_count_with_prefix([counts[j] - (1 if j == i-1 else 0) for j in range(N)])
                 if counts[i-1] > 0 else 0)
                for i in range(1, N + 1)
            ]
            cum_sums = list(reduce(lambda x, y: x + [x[-1] + y] if x else [y], perms, []))
            digit = next(i + 1 for i, s in enumerate(cum_sums) if s >= target_idx)
            prev_sum = cum_sums[digit-2] if digit > 1 else 0
            
            new_counts = [counts[j] - (1 if j == digit-1 else 0) for j in range(N)]
            return ([new_counts, target_idx - prev_sum], state[1] + [digit])

        result = reduce(folder, range(N * K), ([ [K]*N, target ], []))
        print(*(result[1]))

    solve_fixed()

if __name__ == "__main__":
    solve()