import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    # Since S can be enormous, we cannot calculate it directly.
    # However, we can determine the sequence element by element.
    # For the current position, we try numbers v = 1, 2, ..., N.
    # The number of sequences starting with the current prefix is:
    # (Remaining_Length)! / Product(Remaining_Counts_i!)
    
    # To avoid explicit loops, we use reduce to iterate through the positions of the sequence.
    # State: (current_counts, target_index)
    # current_counts: list of remaining counts for each number 1...N
    # target_index: the rank we are looking for
    
    # Precompute factorials is not feasible for N*K = 250,000, 
    # but we only need the ratio of counts.
    # The number of ways to arrange the remaining elements is:
    # W = (sum(counts))! / prod(counts!)
    # If we pick number v, the number of ways is:
    # W_v = (sum(counts) - 1)! / (counts[v-1]-1)! * prod(counts[i]! for i != v-1)
    # W_v = W * counts[v-1] / sum(counts)

    # Because we need the middle sequence, and the total number of sequences S is symmetric,
    # the "middle" sequence is effectively the one that balances the lexicographical distribution.
    # Actually, the problem asks for the floor((S+1)/2)-th sequence.
    # For N=2, K=2, S=6, target=3. Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1)... 3rd is (1,2,2,1).
    
    # Since we cannot use loops, we use reduce over the range of the total length NK.
    # We maintain (current_counts, target_rank, result_sequence).
    
    # To handle the massive numbers, we use Python's arbitrary precision integers.
    # We need a way to calculate the number of permutations of a multiset.
    # Total permutations = (sum(counts))! / prod(counts!)
    
    # Optimization: Instead of calculating full factorials, we can maintain the 
    # current total permutations and multiply by (count[v]/total_remaining).
    
    # Initial total S = (N*K)! / (K!)^N
    # We can't calculate S directly without a loop/recursion for the factorial, 
    # but we can use a math-based approach or a comprehension.
    
    import math
    
    # Using a helper to calculate multiset permutations
    def count_permutations(counts):
        total = sum(counts)
        # Using math.comb or factorial. Since we can't use loops, 
    # we use a functional approach to calculate the denominator.
        denom = reduce(lambda x, y: x * math.factorial(y), counts, 1)
        return math.factorial(total) // denom

    # Initial state
    initial_counts = [K] * N
    # S = count_permutations(initial_counts)
    # target = (S + 1) // 2
    
    # To avoid loops, we use reduce to build the sequence.
    # The state is (counts, target, sequence)
    # We need the total S first.
    S = count_permutations(initial_counts)
    target = (S + 1) // 2
    
    # We use reduce to simulate the process of picking the i-th character.
    # range(N * K) provides the iterations.
    final_state = reduce(
        lambda state, _: (
            (
                # Find the smallest v such that sum of permutations for 1..v >= target
                # We use a list comprehension to calculate cumulative counts for v in 1..N
                # and then find the first v that satisfies the condition.
                # Since we can't use loops, we use a nested reduce or a comprehension 
                # to find the value of v and the updated target.
                (lambda v_found, new_target, new_counts: (new_counts, new_target, state[2] + [v_found]))(
                    # This inner part finds v
                    # We calculate the number of permutations for each possible next digit v
                    # ways(v) = total_ways * counts[v-1] / total_remaining
                    # We find v such that sum_{i=1}^{v-1} ways(i) < target <= sum_{i=1}^{v} ways(i)
                    # Let's use a helper logic inside a comprehension:
                    # We create a list of (v, ways_v) and use reduce to find the cutoff.
                    (lambda options: (
                        # options is [(v, ways), ...]
                        # We need the first v where prefix_sum >= target
                        # We can use a custom reduce to find this v
                        reduce(
                            lambda acc, opt: acc if acc[0] != -1 else (opt[0], target - acc[1], opt[1]) 
                            if opt[1] < target else (opt[0], target - opt[1], opt[1]),
                            options,
                            (-1, target, 0)
                        )
                        # The above reduce is tricky. Let's simplify:
                        # Use a comprehension to get cumulative sums, then find index.
                        # But we can't use loops. We can use map/filter.
                    ))(
                        # Generate (v, ways_v) for v in 1..N if counts[v-1] > 0
                        # ways_v = count_permutations(counts_after_picking_v)
                        [
                            (v, count_permutations([counts[i] - (1 if i == v-1 else 0) for i in range(N)]))
                            for v in range(1, N + 1) if state[0][v-1] > 0
                        ]
                    )
                )
        ),
        range(N * K),
        (initial_counts, target, [])
    )
    
    # The above logic is getting complex for a single reduce. 
    # Let's refine the state transition to be cleaner.
    # Since the prompt forbids loops, I will use a helper function and map/reduce.
    
def solve_final():
    import sys
    import math
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)

    def get_ways(counts):
        return math.factorial(sum(counts)) // reduce(lambda x, y: x * math.factorial(y), counts, 1)

    # State: (counts, target, result)
    # We need to find v such that sum_{i=1}^{v-1} ways(i) < target <= sum_{i=1}^{v} ways(i)
    # Then target = target - sum_{i=1}^{v-1} ways(i)
    
    # To find v without a loop, we can use a list comprehension to calculate ways for all v,
    # then use another comprehension or reduce to find the one.
    
    def step(state, _):
        counts, target, res = state
        # Calculate ways for each possible next digit v (1 to N)
        # only for v where counts[v-1] > 0
        options = [
            (v, get_ways([counts[i] - (1 if i == v-1 else 0) for i in range(N)]))
            for v in range(1, N + 1) if counts[v-1] > 0
        ]
        
        # Find v and the new target
        # We use reduce to iterate through options and find the first one that exceeds target
        # acc = (found_v, current_target, updated_counts)
        result = reduce(
            lambda acc, opt: acc if acc[0] != -1 else 
                (opt[0], target - acc[1], [counts[i] - (1 if i == opt[0]-1 else 0) for i in range(N)])
                if opt[1] < acc[1] else (opt[0], acc[1], [counts[i] - (1 if i == opt[0]-1 else 0) for i in range(N)]),
            options,
            (-1, target, None)
        )
        # The logic above is slightly flawed. Let's use a simpler approach to find v:
        # 1. Calculate cumulative sums of ways.
        # 2. Find the first index where cumsum >= target.
        
        # Correcting the step function logic:
        # We need to find v such that:
        # ways(1) + ... + ways(v-1) < target <= ways(1) + ... + ways(v)
        
        # Let's use a different approach for the step:
        # We'll use a helper to find the index.
        return None # Placeholder for the actual logic implemented below

# Because the constraint to avoid loops is strict, 
# and the logic for finding the lexicographical rank requires 
# iterative state updates, the most "functional" way in Python 
# is using reduce() to simulate the loop and list comprehensions for internal iterations.

def final_implementation():
    import sys
    import math
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)

    def get_ways(counts):
        return math.factorial(sum(counts)) // reduce(lambda x, y: x * math.factorial(y), counts, 1)

    S = get_ways([K] * N)
    target = (S + 1) // 2

    # State: (counts, target, result_list)
    # We use reduce to iterate NK times.
    final_state = reduce(
        lambda state, _: (
            (lambda options: (
                # options: [(v, ways), ...]
                # We need the first v where sum of ways >= target
                # We use a second reduce to find this v and the remaining target
                (lambda found: (
                    found[1], # new_counts
                    found[2], # new_target
                    state[2] + [found[0]] # new_result
                ))(
                    reduce(
                        lambda acc, opt: acc if acc[0] != -1 else 
                        (opt[0], [state[0][i] - (1 if i == opt[0]-1 else 0) for i in range(N)], acc[1] - opt[1])
                        if opt[1] < acc[1] else (opt[0], [state[0][i] - (1 if i == opt[0]-1 else 0) for i in range(N)], acc[1]),
                        options,
                        (-1, None, target)
                    )
                )
            ))(
                [(v, get_ways([state[0][i] - (1 if i == v-1 else 0) for i in range(N)])) 
                 for v in range(1, N + 1) if state[0][v-1] > 0]
            )
        ),
        range(N * K),
        ([K] * N, target, [])
    )
    
    print(*(final_state[2]))

# The logic in the reduce was slightly recursive in thought. 
# Let's provide the clean, working version.