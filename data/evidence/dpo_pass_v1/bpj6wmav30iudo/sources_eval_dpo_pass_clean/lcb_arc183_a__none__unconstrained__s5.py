import sys
from functools import reduce
from math import factorial

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
    # Since N, K <= 500, NK <= 250,000. Factorials are huge but Python handles them.
    # However, we only need the ratio of the total permutations.
    
    # Total permutations of remaining elements:
    # Total! / (c1! * c2! * ... * cN!)
    
    # We use a helper to calculate permutations of a multiset
    # Instead of full factorials every time, we observe:
    # If we pick element 'i', the number of ways to arrange the rest is:
    # (Total-1)! / (c1! ... (ci-1)! ... cN!)
    
    # target is the 1-based index of the sequence
    S = factorial(N * K) // (factorial(K)**N)
    target = (S + 1) // 2

    # We use reduce to iterate NK times. 
    # The accumulator is (counts, target, result_sequence)
    # counts is a tuple to be hashable/immutable if needed, though a list is fine inside reduce.
    
    initial_counts = tuple([K] * N)
    
    def get_count(counts):
        total = sum(counts)
        # Multinomial coefficient: total! / product(c!)
        # We use a property: total! / product(c!)
        # Since we only need this for the current state:
        res = factorial(total)
        # Using a list comprehension and reduce to calculate product of factorials
        denom = reduce(lambda x, y: x * y, [factorial(c) for c in counts], 1)
        return res // denom

    # To avoid O(NK * N) factorial calculations, we can optimize.
    # But with N, K = 500, we must be careful. 
    # Actually, the number of ways to complete the sequence if we pick 'i' is:
    # Ways(counts) * (counts[i-1] / total)
    
    # We'll use a list for counts and modify it. 
    # Since we can't use loops, we use reduce over range(N*K).
    
    # To handle the logic:
    # For each position in the sequence:
    #   For each candidate value v from 1 to N:
    #     If counts[v-1] > 0:
    #       ways = get_count(counts_after_picking_v)
    #       if target <= ways:
    #         pick v, break
    #       else:
    #         target -= ways
    
    # Since we can't use loops, we use a nested reduce or a complex list comprehension.
    # This is tricky. Let's use a helper function and reduce.
    
    def pick_element(state):
        counts, target = state
        total = sum(counts)
        
        # We need to find the smallest v such that the sum of ways for 1..v is >= target
        # We can use a list comprehension to calculate ways for each v
        # ways_for_v = [ (get_count(tuple(counts[:i] + (counts[i]-1,) + counts[i+1:])) 
        #                if counts[i] > 0 else 0 ) for i in range(N) ]
        
        # To avoid O(N) factorial calls per step, we use the formula:
        # ways_for_v = (total_ways * counts[i]) // total
        total_ways = get_count(counts)
        
        # We need to find v such that:
        # sum(ways for 1..v-1) < target <= sum(ways for 1..v)
        
        # We can use a list comprehension to find the prefix sums of ways
        # But we can't use a loop to subtract from target.
        # Instead, we can determine v by comparing target against the prefix sums.
        
        # ways_per_val[i] is the number of sequences starting with value i+1
        ways_per_val = [( (total_ways * counts[i]) // total ) if counts[i] > 0 else 0 for i in range(N)]
        
        # prefix_sums[i] is the sum of ways for values 1..i+1
        # Using reduce to build prefix sums is possible but messy. 
        # We can use a list comprehension with sum().
        # v is the first index where sum(ways_per_val[:v+1]) >= target
        
        # Since we can't use a loop, we find v using a list comprehension and next()
        v = next(i for i in range(N) if sum(ways_per_val[:i+1]) >= target)
        
        # Update target for the next position:
        # new_target = target - sum(ways_per_val[:v])
        new_target = target - sum(ways_per_val[:v])
        
        # Update counts:
        new_counts = tuple(counts[:v] + (counts[v]-1,) + counts[v+1:])
        
        return (new_counts, new_target, v + 1)

    # Use reduce to generate the sequence
    # The state is (counts, target, result_list)
    final_state = reduce(
        lambda state, _: (
            # We need to call pick_element. 
            # Since we need to update the state, we'll use a helper.
            # But we can't define functions inside reduce easily.
            # Let's embed the logic.
            (lambda res: (res[0], res[1], state[2] + [res[2]]))(
                (lambda cnts, tgt: (
                    (lambda tw, tot: (
                        (lambda ways: (
                            (lambda v: (
                                tuple(cnts[:v] + (cnts[v]-1,) + cnts[v+1:]),
                                tgt - sum(ways[:v]),
                                v + 1
                            ))(next(i for i in range(N) if sum(ways[:i+1]) >= tgt))
                        ))(( (tw * or_zero(cnts[i])) // tot if cnts[i] > 0 else 0 for i in range(N) ))
                    ))(
                        get_count(cnts), sum(cnts)
                    ))(state[0], state[1])
                )
            )
        ),
        range(N * K),
        (initial_counts, target, [])
    )
    
    # The logic above is getting recursive/nested. Let's simplify.
    # Since I cannot use loops, I will use a recursive-like structure 
    # via reduce and a helper function defined outside.
    pass

# Redefining solve to be cleaner and strictly follow rules
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    # Precompute factorials for speed
    # Using a list comprehension to create a factorial lookup table
    # Since we can't use loops, we use reduce to build the factorial list
    fact = reduce(lambda acc, i: acc + [acc[-1] * i], range(1, N * K + 1), [1])
    
    def get_multinomial(counts):
        total = sum(counts)
        denom = reduce(lambda x, y: x * y, [fact[c] for c in counts], 1)
        return fact[total] // denom

    # State: (counts, target, result)
    # We use a helper function to process one step
    def step(state, _):
        counts, target, result = state
        total_ways = get_multinomial(counts)
        total_elements = sum(counts)
        
        # ways_per_val[i] = ways to complete if we pick value i+1
        # ways = (total_ways * counts[i]) // total_elements
        ways_per_val = [( (total_ways * counts[i]) // total_elements ) if counts[i] > 0 else 0 for i in range(N)]
        
        # Find v: the first index where prefix sum >= target
        # We use a generator expression with next()
        v = next(i for i in range(N) if sum(ways_per_val[:i+1]) >= target)
        
        new_counts = list(counts)
        new_counts[v] -= 1
        return (tuple(new_counts), target - sum(ways_per_val[:v]), result + [v + 1])

    S = fact[N * K] // (fact[K]**N)
    target = (S + 1) // 2
    
    final_result = reduce(step, range(N * K), (tuple([K] * N), target, []))
    print(*(final_result[2]))

if __name__ == "__main__":
    solve_final()