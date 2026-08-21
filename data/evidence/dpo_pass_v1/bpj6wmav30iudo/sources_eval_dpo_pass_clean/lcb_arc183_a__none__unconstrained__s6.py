The constraint to avoid explicit loops and recursion forces a functional programming style, which often encourages the use of immutable data structures and high-level abstractions. However, for a problem requiring the construction of a sequence based on combinatorial counts (specifically multinomial coefficients), it challenges the idiomatic nature of Python, which typically favors `for` loops for stateful sequence generation. To solve this, I will use `functools.reduce` to simulate the iterative process of filling the sequence and list comprehensions to handle the selection of the next element.

```python
import sys
from math import factorial
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The total number of good sequences S is the multinomial coefficient:
    # S = (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2
    
    # To avoid massive memory usage and recursion, we use reduce to build the sequence.
    # The state is (current_counts, target_index)
    # current_counts: a list of how many of each number 1..N are left to place.
    
    # Precompute factorials for the multinomial formula
    # Since N, K <= 500, NK <= 250,000. 
    # We use a lambda to calculate the number of permutations of the remaining elements.
    # Permutations = (sum(counts))! / product(counts[i]!)
    
    # Note: Calculating factorial(250000) is possible in Python but slow.
    # However, we only need to compare the target index with the number of sequences 
    # starting with 1, then 2, etc.
    # Let C be the total remaining slots. If we pick number 'i', the number of ways 
    # to arrange the rest is (C-1)! / (K1! * ... * (Ki-1)! * ... * KN!)
    
    # To optimize, we observe that the number of ways to arrange the rest given we pick 'i'
    # is: (Total ways to arrange all remaining) * (count[i] / Total remaining)
    
    # Let's define the logic inside reduce.
    # state: (counts, target, result_sequence)
    
    # Because we cannot use loops, we use a list comprehension to find the 
    # smallest 'i' such that the sum of counts of sequences starting with 1..i 
    # is >= target.
    
    # Since we need the (S+1)//2 - th sequence, and S can be huge, 
    # we use Python's arbitrary precision integers.
    
    # Total S = factorial(N*K) // (factorial(K)**N)
    # target = (S + 1) // 2
    
    # We use a helper to calculate the number of permutations of the current multiset.
    # Since we can't define complex functions with loops, we use a lambda.
    # ways = factorial(sum(counts)) // reduce(lambda a, b: a * factorial(b), counts, 1)
    
    # To avoid redundant factorial calls, we can't really "memoize" without a loop/recursion,
    # but we can use the property: 
    # Ways(picking i) = Ways(total) * counts[i] / total_remaining
    
    # Initial values
    initial_counts = [K] * N
    total_S = factorial(N * K) // (factorial(K)**N)
    initial_target = (total_S + 1) // 2
    
    # We use reduce to iterate NK times.
    # The range(N * K) acts as our loop.
    # The accumulator is (counts, target, sequence)
    
    final_state = reduce(
        lambda state, _: (
            # Calculate the index of the element to pick
            # We need the smallest i such that:
            # sum_{j=1}^{i} (Ways(total) * counts[j-1] / total_remaining) >= target
            (lambda counts, target, total_rem, total_ways: (
                # Find i (1-indexed)
                # We use a list comprehension to calculate the cumulative counts
                # and then find the first index where it exceeds target.
                (lambda cumulative_ways: (
                    # i is the index of the first element in cumulative_ways >= target
                    # We use a list comprehension to find all indices that satisfy this, 
                    # then take the min.
                    [
                        (i + 1, counts[i]) 
                        for i, cw in enumerate(cumulative_ways) if cw >= target
                    ][0]
                ))(
                    # cumulative_ways[i] = sum_{j=0}^{i} (total_ways * counts[j] // total_rem)
                    # We can't use a loop, so we use another reduce or a clever comprehension.
                    # Actually, we can just calculate the ways for each i in a list.
                    # ways_for_i = (total_ways * counts[i]) // total_rem
                    # Since we need cumulative, we can use a list comprehension with sum()
                    # but that's O(N^2). With N=500, that's acceptable per step? 
                    # No, NK * N = 250,000 * 500 = 125 million. Too slow.
                    # But wait, we only need to find the one 'i'.
                    # We can use a list comprehension to find the 'i' by 
                    # iterating through the counts.
                    # Since we can't use a loop, we'll use a list comprehension 
                    # to generate the sequence of cumulative sums.
                    # To avoid O(N^2), we can't. But we can use a 
                    # mathematical approach to find i.
                    
                    # Let's use a list comprehension to find the i:
                    # we want the smallest i such that sum_{j=0}^{i} (total_ways * counts[j] // total_rem) >= target
                    # We can pre-calculate the "unit" way: unit = total_ways // total_rem
                    # then we need sum_{j=0}^{i} (unit * counts[j]) >= target
                    # i.e., sum_{j=0}^{i} counts[j] >= target / unit
                    
                    # Let unit = total_ways // total_rem
                    # target_count = (target + unit - 1) // unit
                    # Then we find i such that sum of counts up to i is >= target_count.
                    
                    # Let's implement this:
                    (lambda unit, target_count: (
                        # Find the first i where sum(counts[:i+1]) >= target_count
                        # We can use a list comprehension to find all such i, then min.
                        # To avoid O(N^2), we can't use sum(counts[:i+1]) inside.
                        # But we can use a trick: 
                        # Since we only need the result, and N is 500, 
                        # maybe we can just use a list comprehension to 
                        # calculate the cumulative sums once per outer step.
                        # cumulative = list(itertools.accumulate(counts))
                        # But we can't import itertools.
                        # We can use reduce to build the cumulative list.
                        (lambda cum_counts: (
                            # Find the first index where cum_counts[i] >= target_count
                            [i for i, val in enumerate(cum_counts) if val >= target_count][0]
                        ))(
                            # Simulate accumulate using reduce
                            reduce(lambda acc, x: acc + [acc[-1] + x] if acc else [x], counts, [])
                        )
                    ))(
                        total_ways // total_rem,
                        (target + (total_ways // total_rem) - 1) // (total_ways // total_rem)
                    )
                )
            )(
                state[0], state[1], sum(state[0]), 
                # total_ways for the current state
                # total_ways = factorial(sum(counts)) // product(factorial(c))
                # We can't use a loop for product, so we use reduce.
                factorial(sum(state[0])) // reduce(lambda a, b: a * factorial(b), state[0], 1)
            )
        ),
        range(N * K),
        (initial_counts, initial_target, [])
    )

    # The reduce logic above is a bit messy because of the constraints.
    # Let's refine it into a clean structure.
    # We need to return the sequence.
    # Since I cannot use loops, I will use a recursive-like structure 
    # implemented via reduce on a range.
    
    # Correct logic for target update:
    # target = target - (sum of ways for elements < chosen i)
    # target = target - (cumulative_ways[i-1])
    
    # Let',s redefine the reduce function clearly.
    pass

# Because of the extreme constraints and the nature of the problem (state updates),
# the most reliable way to implement this without loops/recursion is 
# to use a list comprehension to handle the "loop" via a custom 
# reduction that passes the state forward.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    # S = (NK)! / (K!)^N
    # target = (S + 1) // 2
    # We use reduce to simulate:
    # for _ in range(N*K):
    #   total_ways = factorial(sum(counts)) // product(factorial(c))
    #   unit = total_ways // sum(counts)
    #   # find i such that sum(counts[:i]) < target <= sum(counts[:i+1]) * unit
    #   # ... update counts, target, sequence
    
    # To avoid O(N^2) or O(N*NK), we must be careful.
    # Actually, N=500, NK=250,000. O(NK * N) is too slow.
    # But we only need to find 'i'. 
    # target_count = (target + unit - 1) // unit
    # we need smallest i such that sum(counts[:i+1]) >= target_count.
    
    # Let',s use the fact that we only need to track the 'target_count'.
    # The "unit" is the number of ways to arrange the remaining elements 
    # given one specific element is chosen.
    # unit = (sum(counts)-1)! / product(counts_after_picking_i!)
    
    # Let's use a simpler approach for the target:
    # In each step, we want to pick the smallest x in {1..N} such that:
    # (Ways to arrange remaining if we pick x) >= target
    # If not, we subtract those ways from target and try x+1.
    
    # Ways(x) = (TotalRemaining - 1)! / (K1! * ... * (Kx-1)! * ... * KN!)
    # Ways(x) = [ (TotalRemaining)! / (K1! * ... * KN!) ] * Kx / TotalRemaining
    
    # Let TotalWays(state) = (sum(counts))! / product(counts!)
    # Ways(x) = TotalWays(state) * counts[x-1] / sum(counts)
    
    # To avoid O(N*NK), we observe that we only need to know 
    # how many elements of the current counts we skip.
    # target_count = (target * sum(counts) + TotalWays(state) - 1) // TotalWays(state)
    # This is not quite right. 
    # Let W = TotalWays(state). The number of sequences starting with 1 is W * counts[0] / sum(counts).
    # The number of sequences starting with 2 is W * counts[1] / sum(counts).
    # We pick i such that:
    # sum_{j=0}^{i-2} (W * counts[j] / sum(counts)) < target <= sum_{j=0}^{i-1} (W * counts[j] / sum(counts))
    
    # Let,s use the target_count logic:
    # target_count = (target * sum(counts) + W - 1) // W
    # Then we find i such that sum(counts[:i-1]) < target_count <= sum(counts[:i])
    
    # Since we can't use loops, we use reduce.
    # state = (counts, target, sequence)
    
    # Precompute factorials to avoid redundant work
    # But we can't have a loop to precompute. 
    # We can use a list comprehension:
    # facts =