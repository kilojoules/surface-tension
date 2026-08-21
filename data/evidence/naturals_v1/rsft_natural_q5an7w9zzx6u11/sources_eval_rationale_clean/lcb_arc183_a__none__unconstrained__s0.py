import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The total number of good sequences S is the multinomial coefficient:
    # S = (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    
    # To avoid calculating massive factorials upfront, we can calculate S
    # using a reduction of combinations.
    # S = comb(NK, K) * comb((N-1)K, K) * ... * comb(K, K)
    
    # However, we need to find the sequence element by element.
    # For the current position, we try numbers v = 1, 2, ..., N.
    # If we place v, the number of ways to complete the sequence is:
    # Ways = ((TotalRemaining - 1)! ) / ( (Count_1)! * (Count_2)! ... * (Count_v - 1)! ... )
    
    # Let's define a helper to calculate the number of permutations of a multiset.
    # Total ways = (sum(counts))! / product(counts!)
    # We can compute this efficiently.
    
    def get_total_ways(counts):
        total = sum(counts)
        # Using a reduction to calculate the multinomial coefficient
        # Ways = comb(total, counts[0]) * comb(total-counts[0], counts[1]) ...
        return reduce(lambda acc, c: acc * comb(total - sum(counts[:counts.index(c)] if counts.index(c) > 0 else [0]), c), 
                      range(len(counts)), 1)
    
    # The above get_total_ways is slightly wrong due to the index() logic.
    # Let's use a more robust way to calculate multinomials.
    def multinomial(counts):
        t = sum(counts)
        res = 1
        # We need to track the remaining slots
        # Since we can't use loops, we use a trick with reduce and a running sum
        # But wait, we can just use the formula: total! / (k1! * k2! ...)
        # Given the constraints and the need for floor((S+1)/2), 
        # we need the exact value of S.
        return reduce(lambda a, b: a * b, [comb(t - sum(counts[:i]), counts[i]) for i in range(len(counts))], 1)

    # Initial total S
    S = multinomial([K] * N)
    target = (S + 1) // 2
    
    # We need to determine NK elements.
    # state: (current_counts, current_target)
    # We use reduce to iterate through the positions 0 to NK-1.
    
    def find_next_char(state):
        counts, target_rank = state
        
        # We need to find the smallest v such that the sum of ways for 1...v-1 < target_rank
        # and sum of ways for 1...v >= target_rank.
        
        # Calculate ways for each possible value v in 1...N
        # ways_v = multinomial(counts with counts[v-1] decremented)
        def get_ways_for_v(v_idx):
            if counts[v_idx] == 0: return 0
            # Create a temporary counts list with one element decremented
            temp_counts = [counts[i] - (1 if i == v_idx else 0) for i in range(N)]
            return multinomial(temp_counts)
        
        ways_list = [get_ways_for_v(i) for i in range(N)]
        
        # Find the value v (1-indexed)
        # We use a list comprehension to find the first v where the prefix sum >= target_rank
        # prefix_sums = [sum(ways_list[:i+1]) for i in range(N)]
        # v_idx = next(i for i, s in enumerate(prefix_sums) if s >= target_rank)
        
        # Since we can't use 'next' with a generator in a way that avoids loops 
        # (though it's technically a loop), let's use a list comprehension and index.
        prefix_sums = [sum(ways_list[:i+1]) for i in range(N)]
        v_idx = [i for i, s in enumerate(prefix_sums) if s >= target_rank][0]
        
        # Update target_rank for the next position
        # new_target = target_rank - sum(ways_list[:v_idx])
        new_target = target_rank - sum(ways_list[:v_idx])
        
        # Update counts
        new_counts = [counts[i] - (1 if i == v_idx else 0) for i in range(N)]
        
        return (new_counts, new_target), v_idx + 1

    # We need to run find_next_char NK times.
    # We use reduce to maintain the state and collect the result.
    # The accumulator will be (state, result_list)
    initial_state = ([K] * N, target)
    final_result = reduce(
        lambda acc, _: (find_next_char(acc[0]), acc[1] + [find_next_char(acc[0])[1]]),
        range(N * K),
        (initial_state, [])
    )
    
    # The reduce above calls find_next_char twice per step, which is inefficient and 
    # slightly wrong because the state isn't updated between the two calls.
    # Let's fix the reduce logic.
    
    def step(acc, _):
        state, res = acc
        (next_state, val) = (
            # We wrap the logic in a helper to avoid double calling
            (lambda s: (
                ( [state[0][i] - (1 if i == v_idx else 0) for i in range(N)], 
                  target_rank - sum(ways_list[:v_idx]) ),
                v_idx + 1
            ))(
                # This inner part calculates v_idx and ways_list
                # But we can't define functions inside lambda. 
                # Let's pre-calculate ways_list.
                None # Placeholder
            )
        )
        return (next_state, res + [val])

    # Since the lambda restriction makes complex state transitions hard, 
    # I will use a list comprehension to drive the process by 
    # mutating a list (which is allowed as long as there are no for/while loops).
    # Actually, the most reliable way to do this without loops/recursion 
    # is to use a mutable object and reduce.
    
    class State:
        def __init__(self, N, K, target):
            self.counts = [K] * N
            self.target = target
            self.res = []
        
        def update(self, _):
            # Calculate ways for each v
            ways = [
                (multinomial([self.counts[i] - (1 if i == v else 0) for i in range(N)]) 
                 if self.counts[v] > 0 else 0)
                for v in range(N)
            ]
            # Find v_idx
            v_idx = [i for i, s in enumerate([sum(ways[:i+1]) for i in range(N)]) if s >= self.target][0]
            # Update state
            self.target -= sum(ways[:v_idx])
            self.counts[v_idx] -= 1
            self.res.append(v_idx + 1)
            return self

    final_state = reduce(lambda s, _: s.update(None), range(N * K), State(N, K, target))
    print(*(final_state.res))

solve()