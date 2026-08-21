import sys
from math import factorial
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
    # Let target = (S + 1) // 2
    
    # To avoid calculating S explicitly (which is huge), 
    # we can determine the sequence element by element.
    # For the first position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with v is (NK-1)! / ((K-1)! * (K!)^{N-1})
    # which simplifies to S * (K / NK).
    
    # Let's use a recursive-like approach via reduce to maintain state:
    # State: (current_counts, target_index)
    # current_counts: list of remaining counts for each number 1...N
    # target_index: the rank we are looking for (1-indexed)

    def get_count(counts):
        # Total permutations of multiset: (sum(counts))! / product(counts[i]!)
        total = sum(counts)
        # We use a property: Total = (sum counts)! / prod(c!)
        # This can be very large, so we handle it with Python's arbitrary precision ints.
        res = factorial(total)
        for c in counts:
            res //= factorial(c)
        return res

    # Initial state
    initial_counts = [K] * N
    
    # Calculate S = get_count(initial_counts)
    S = get_count(initial_counts)
    target = (S + 1) // 2

    # We use reduce to iterate through all NK positions
    # range(N * K) acts as the loop counter
    # accumulator: (counts, target, result_sequence)
    
    def step(acc, _):
        counts, target, res = acc
        
        # Find the smallest v such that the sum of sequences starting with 1...v 
        # is >= target
        # We search for v in 1...N
        
        def find_v(v, current_target):
            if counts[v-1] > 0:
                # Count of sequences if we pick v
                # New counts would be counts with counts[v-1] decremented
                # Instead of full factorial, we can derive it from the total count of the remaining slots
                # Count = (Total-1)! / (counts[0]! ... (counts[v-1]-1)! ... counts[N-1]!)
                # Count = [Total! / prod(counts!)] * (counts[v-1] / Total)
                
                # However, calculating the total for the remaining suffix is safer:
                # We need a way to calculate the number of ways to fill the rest.
                # Let's use a helper to calculate permutations of the remaining multiset.
                
                # To avoid explicit loops, we can't use 'for v in range(1, N+1)'
                # But we can use a recursive search or a list comprehension with next()
                pass

        # Since we cannot use loops, we use a generator expression with next() 
        # to find the first v that satisfies the condition.
        
        # To calculate the number of sequences starting with v:
        # ways(v) = get_count(counts_after_picking_v)
        
        # We need to find v such that sum_{i=1}^{v-1} ways(i) < target <= sum_{i=1}^{v} ways(i)
        
        # We can use a helper function to iterate through v using recursion or map
        def search_v(v, accumulated_ways):
            if v > N: return None # Should not happen
            
            # Calculate ways if we pick v
            if counts[v-1] > 0:
                # Number of ways to arrange the remaining items
                # Total remaining items = sum(counts) - 1
                # Ways = (sum-1)! / (counts[0]! ... (counts[v-1]-1)! ... counts[N-1]!)
                # This is equivalent to: (Total_Ways * counts[v-1]) // Total_Items
                
                # Let's just call get_count with a modified list
                # To avoid loops, we create the modified list using slicing
                temp_counts = counts[:v-1] + [counts[v-1]-1] + counts[v:]
                ways_v = get_count(temp_counts)
                
                if accumulated_ways + ways_v >= target:
                    return (v, target - accumulated_ways, temp_counts)
                else:
                    return search_v(v + 1, accumulated_ways + ways_v)
            else:
                return search_v(v + 1, accumulated_ways)

        v, new_target, next_counts = search_v(1, 0)
        return (next_counts, new_target, res + [v])

    final_state = reduce(step, range(N * K), (initial_counts, target, []))
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()