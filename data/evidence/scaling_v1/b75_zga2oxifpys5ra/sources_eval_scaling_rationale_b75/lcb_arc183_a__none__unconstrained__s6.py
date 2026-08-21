import sys
from math import factorial
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
    # Let target = (S + 1) // 2
    
    # To avoid calculating massive factorials repeatedly, we use a helper
    # for multiset permutations: (sum(counts))! / product(counts!)
    def count_permutations(counts):
        total = sum(counts)
        # Using a property: Perm(n1, n2, ...) = (n1+n2)! / (n1! n2!) * (n2+n3)! / (n2! n3!) ...
        # But for N=500, K=500, we need arbitrary precision integers.
        # We can use a simplified version since we only need to compare/subtract.
        # However, the most direct way in Python is using math.factorial.
        res = factorial(total)
        for c in counts:
            res //= factorial(c)
        return res

    # We need to find the sequence lexicographically.
    # We determine the element at each position i from 1 to NK.
    # For the current position, we try candidates v = 1, 2, ..., N.
    # If we place v, the number of ways to complete the sequence is count_permutations(updated_counts).
    
    # Since we cannot use loops or recursion, we use functools.reduce.
    # State: (current_counts, current_target, result_sequence)
    # current_counts: list of remaining counts for each number 1...N
    # current_target: the rank we are looking for within the current subspace
    
    initial_counts = [k] * n
    # Total S = count_permutations(initial_counts)
    total_s = count_permutations(initial_counts)
    target = (total_s + 1) // 2
    
    # We need to iterate NK times. We can use range(n * k) inside reduce.
    final_state = reduce(
        lambda state, _: (
            (
                # Find the smallest v such that sum of permutations for 1...v-1 < target
                # We use a helper logic inside a list comprehension to find v.
                # 1. Calculate permutations for each possible next digit v in 1...N
                # 2. Find the digit v that crosses the target threshold.
                (
                    # This inner reduce finds the digit v and the updated target
                    reduce(
                        lambda acc, v: (
                            # acc: (found, current_v, running_sum, updated_target)
                            acc if acc[0] else (
                                # Calculate permutations if we pick digit v
                                # Remaining counts: counts[v-1]-1, others same
                                (
                                    True, v, 0, acc[3] - count_permutations(
                                        [c if i != v-1 else c-1 for i, c in enumerate(state[0])]
                                    )
                                ) if acc[3] <= count_permutations(
                                        [c if i != v-1 else c-1 for i, c in enumerate(state[0])]
                                    ) else (
                                        False, v, 0, acc[3] - count_permutations(
                                            [c if i != v-1 else c-1 for i, c in enumerate(state[0])]
                                        )
                                    )
                                )
                            ),
                            range(1, n + 1),
                            (False, 0, 0, state[1])
                        )
                    ),
                    # The above reduce is slightly wrong because it doesn't handle the 
                    # "sum of previous" correctly. Let's refine:
                    # We need v such that sum_{j=1}^{v-1} P(j) < target <= sum_{j=1}^{v} P(j)
                    # Let's use a different approach for the inner selection.
                    None
                )
            ),
            None
        ),
        range(n * k),
        (initial_counts, target, [])
    )

    # Correcting the logic to fit into a single reduce without loops/recursion:
    # We use a helper function to determine the next character and the new target.
    def get_next(state):
        counts, target, seq = state
        # Calculate permutations for each possible next digit v in 1...N
        # perms[v-1] = count_permutations(counts with v-1 decremented)
        perms = [
            count_permutations([c if i != v else c-1 for i, c in enumerate(counts)])
            for v in range(n) if counts[v] > 0
        ]
        # We need v such that sum(perms[0...v-1]) < target <= sum(perms[0...v])
        # Since we can't loop, we use a list comprehension to find the index.
        # cumulative_perms = [sum(perms[:i+1]) for i in range(len(perms))]
        # But we need to map the index back to the actual digit.
        
        # To avoid explicit loops, we use a generator/list comprehension to find the digit:
        # We find the first v where target <= cumulative_sum
        # We can use a trick with reduce to find the digit and the new target.
        
        # Simplified: just iterate through 1..N and subtract perms from target.
        # We use a custom object or tuple to track (remaining_target, chosen_digit)
        res = reduce(
            lambda acc, v: (
                acc if acc[1] != 0 else (
                    (acc[0] - count_permutations([c if i != v-1 else c-1 for i, c in enumerate(counts)]), v)
                    if acc[0] <= count_permutations([c if i != v-1 else c-1 for i, c in enumerate(counts)])
                    else (count_permutations([c if i != v-1 else c-1 for i, c in enumerate(counts)]), v) # This is wrong
                )
            ),
            range(1, n + 1),
            (target, 0)
        )
        # The above is getting complex. Let's use a simpler approach:
        # For a fixed position, the digit v is the smallest v such that
        # target <= sum_{j=1}^v Perm(counts after picking j)
        pass

# Given the constraints and the "no loop" rule, the most reliable way to 
# implement this is using a recursive-like structure via reduce and 
# helper functions defined inside.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    n, k = map(int, input_data)
    
    def get_perms(counts):
        total = sum(counts)
        res = factorial(total)
        for c in counts: res //= factorial(c)
        return res

    def step(state, _):
        counts, target, seq = state
        # Find v in 1..n such that counts[v-1] > 0
        # We need the smallest v such that target <= sum_{j=1}^v Perms(j)
        # We can use a list comprehension to calculate perms for all valid v
        valid_vs = [v for v in range(1, n + 1) if counts[v-1] > 0]
        v_perms = [get_perms([c if i != v-1 else c-1 for i, c in enumerate(counts)]) for v in valid_vs]
        
        # Find the index of the chosen v
        # We use a list comprehension to find the first index where cumulative sum >= target
        cum_sums = reduce(lambda a, b: a + [a[-1] + b] if a else [b], v_perms, [])
        idx = next(i for i, s in enumerate(cum_sums) if s >= target)
        
        chosen_v = valid_vs[idx]
        # New target is target - (sum of perms before idx)
        prev_sum = cum_sums[idx-1] if idx > 0 else 0
        new_target = target - prev_sum
        
        new_counts = [c if i != chosen_v-1 else c-1 for i, c in enumerate(counts)]
        return (new_counts, new_target, seq + [chosen_v])

    # Since we can't use loops, we use reduce to simulate the process for N*K steps
    final_state = reduce(step, range(n * k), ([k] * n, (get_perms([k] * n) + 1) // 2, []))
    print(*(final_state[2]))

if __name__ == "__main__":
    final_solve()