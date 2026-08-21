import sys
from functools import reduce
from math import factorial

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Since we need a specific index, we can use a greedy approach.
    # For each position, we try numbers 1 to N.
    # The number of ways to complete a sequence given remaining counts c1, c2, ..., cN
    # is (sum(ci))! / product(ci!)
    
    # To avoid explicit loops and recursion, we use reduce to iterate through the 
    # NK positions of the sequence.
    # State: (current_counts, target_index, result_sequence)
    # current_counts: list of remaining counts for each number 1...N
    # target_index: the rank we are looking for (1-indexed)
    
    def get_total_permutations(counts):
        total_len = sum(counts)
        # Using a helper to calculate multiset permutation formula
        # Permutations = (sum(counts))! / product(counts!)
        # We use a generator expression inside reduce to calculate the denominator
        denom = reduce(lambda a, b: a * factorial(b), counts, 1)
        return factorial(total_len) // denom

    # Initial S calculation to find the target rank
    # S = factorial(n*k) // (factorial(k)**n)
    s = factorial(n * k) // (factorial(k)**n)
    target_rank = (s + 1) // 2

    # We use reduce to simulate the process of filling NK slots.
    # The range(n * k) provides the iterations.
    # The accumulator carries (counts, rank, sequence)
    final_state = reduce(
        lambda state, _: (
            # Inside this lambda, we need to determine which number i (1 to N) fits in the current slot.
            # We can't use a loop, so we use another reduce or a list comprehension to find the digit.
            (lambda counts, rank: (
                # Find the smallest i such that the sum of permutations for 1...i-1 is less than rank
                # and the sum for 1...i is >= rank.
                (lambda i_found, new_rank, new_counts: (
                    # Update counts for the chosen digit
                    [c - 1 if idx == i_found - 1 else c for idx, c in enumerate(counts)],
                    new_rank,
                    state[2] + [i_found]
                )) (
                    # This inner logic finds the digit i
                    # We use a helper list to calculate cumulative counts of permutations
                    (lambda options: (
                        # options is a list of (digit, count_of_permutations)
                        # We find the first digit where the cumulative sum reaches the rank.
                        # Since we can't loop, we use a trick with a list comprehension and next()
                        next(
                            (digit for digit, cum_sum in 
                             # Calculate cumulative permutations for digits 1...N
                             # We use a custom scan (prefix sum) implemented via reduce
                             # to find the range the target_rank falls into.
                             (lambda scan: [
                                 (d, s_val) for d, s_val in scan
                              ])(
                                  # This is a complex way to simulate a loop to find the digit
                                  # We map each possible digit to the number of permutations it would head
                                  # then we use a custom reduce to track the cumulative sum.
                                  # However, a simpler way is to use a generator and next()
                                  # by iterating through digits 1...N and subtracting permutations.
                                  # But we need to maintain state, so we use a helper function.
                                  # Wait, the prompt forbids loops/recursion. 
                                  # Let's use a list comprehension to pre-calculate permutation counts for each digit.
                                  # For digit d, if we place d, the remaining permutations are:
                                  # (Total-1)! / ( (c_d - 1)! * product(c_j!) )
                                  # = [Total! / product(c_j!)] * (c_d / Total)
                                  # = Total_Permutations * c_d / Total_Len
                                  [
                                      (d, (get_total_permutations(counts) * counts[d-1]) // sum(counts))
                                      for d in range(1, n + 1) if counts[d-1] > 0
                                  ]
                              )
                        ))
                    )(
                        # We need to find the digit and the updated rank.
                        # We use a helper function defined via a lambda to iterate and subtract.
                        # Since we can't use loops, we use a trick: 
                        # we create a list of (digit, rank_after_subtracting)
                        # and pick the one where the rank became <= 0.
                        (lambda digit_perms: (
                            # digit_perms: [(d1, p1), (d2, p2), ...]
                            # We want the first d_i where sum(p_1...p_{i-1}) < rank <= sum(p_1...p_i)
                            (lambda search: (
                                # search is a list of (digit, current_rank)
                                # We use reduce to propagate the rank subtraction
                                reduce(
                                    lambda acc, x: (x[0], acc[1] - x[1]) if acc[1] > 0 else (acc[0], acc[1]),
                                    digit_perms,
                                    (None, rank)
                                )
                            )) (digit_perms)
                        )) (
                            [(d, (get_total_permutations(counts) * counts[d-1]) // sum(counts)) 
                             for d in range(1, n + 1) if counts[d-1] > 0]
                        )
                    )
                )
            )(state[0], state[1])
        ),
        range(n * k),
        ([k] * n, target_rank, [])
    )

    # The above logic is getting nested. Let's refine it into a clean reduce.
    # We need a way to find the digit and the new rank without a loop.
    # Let's use a helper function for the digit selection logic.
    
    def get_next_digit(counts, rank):
        total_len = sum(counts)
        # Permutations if we pick digit d: (total_len-1)! / ( (c_d-1)! * product(c_j!) )
        # This simplifies to: [ (total_len)! / product(c_j!) ] * c_d / total_len
        total_perms = get_total_permutations(counts)
        
        # Create a list of (digit, perms_for_this_digit)
        options = [(d + 1, (total_perms * counts[d]) // total_len) 
                   for d in range(n) if counts[d] > 0]
        
        # Use reduce to find the digit and the remaining rank
        # acc: (chosen_digit, current_rank)
        res = reduce(
            lambda acc, opt: (opt[0], acc[1] - opt[1]) if acc[0] is None and acc[1] > opt[1] 
                             else acc if acc[0] is not None else (opt[0], acc[1] - opt[1]),
            options,
            (None, rank)
        )
        # The above reduce is slightly wrong. Let's use a different approach:
        # We want the first d where rank <= perms_for_d.
        # If rank > perms_for_d, we subtract perms_for_d and move to next d.
        
        # Correct reduce to find digit and new rank:
        # state: (found_digit, current_rank)
        final = reduce(
            lambda acc, opt: (acc[0], acc[1]) if acc[0] is not None else 
                              (opt[0], acc[1] - opt[1]) if acc[1] > opt[1] 
                              else (opt[0], acc[1]),
            options,
            (None, rank)
        )
        return final[0], final[1]

    # Since the prompt forbids recursion and loops, we must use reduce for the main sequence.
    # We redefine the state transition clearly.
    def step(state, _):
        counts, rank, seq = state
        digit, new_rank = get_next_digit(counts, rank)
        new_counts = [c - 1 if i == digit - 1 else c for i, c in enumerate(counts)]
        return (new_counts, new_rank, seq + [digit])

    result = reduce(step, range(n * k), ([k] * n, target_rank, []))
    print(*(result[2]))

solve()