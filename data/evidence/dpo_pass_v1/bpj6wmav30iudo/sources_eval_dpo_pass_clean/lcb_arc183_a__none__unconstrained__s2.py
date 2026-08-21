import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # Total number of good sequences S is (N*K)! / (K!^N)
    # We want the floor((S+1)/2)-th sequence.
    # S = comb(NK, K) * comb((N-1)K, K) * ... * comb(K, K)
    
    # Precompute total S using reduce
    total_s = reduce(lambda acc, i: acc * comb(i, K), range((N-1)*K, N*K + 1, K), 1)
    target_rank = (total_s + 1) // 2

    # We need to determine the sequence element by element.
    # Let 'counts' be the remaining number of times each digit 1..N can appear.
    # For each position, we try digits d = 1..N.
    # The number of ways to complete the sequence given current counts is:
    # (Sum of counts)! / Product(counts[i]!)
    
    # Since we cannot use loops, we use reduce to iterate through the NK positions.
    # The accumulator will be (current_rank, current_counts, result_sequence).
    
    def get_ways(counts):
        # Multinomial coefficient: (sum(counts))! / product(c!)
        # Calculated as product of comb(sum_so_far, c)
        return reduce(lambda acc, c: acc * comb(sum(counts[counts.index(c):]), c), counts, 1)

    # To avoid the complexity of updating a list in reduce without loops, 
    # we use a tuple for counts.
    initial_counts = tuple([K] * N)
    
    def find_digit(state, _):
        rank, counts, seq = state
        
        # We need to find the smallest d such that the sum of ways for 1..d-1 
        # is less than rank, and ways for 1..d is >= rank.
        # Since we can't loop, we use a list comprehension to calculate ways for each d.
        
        # ways_for_d[d-1] = ways to form sequence if we pick digit d at current position
        # If counts[d-1] == 0, ways = 0.
        # Else, ways = (Total remaining - 1)! / (counts[0]! ... (counts[d-1]-1)! ... counts[N-1]!)
        
        # Simplified: ways = (Total remaining) * (Ways with counts) / counts[d-1]
        # But it's safer to just calculate the multinomial for the remaining slots.
        
        total_rem = sum(counts)
        
        # Calculate ways for each possible digit d (1 to N)
        # We use a list comprehension to evaluate the "cost" of each digit
        digit_ways = [
            (get_ways(tuple(counts[i] - (1 if i == d_idx else 0) for i in range(N))) 
             if counts[d_idx] > 0 else 0)
            for d_idx in range(N)
        ]
        
        # Find the digit d such that the rank falls within its range.
        # We use a generator/next to find the first d where cumulative ways >= rank.
        # Since we can't use loops, we can use a trick with reduce or a 
        # carefully constructed list comprehension.
        
        # cumulative_ways[i] is the sum of ways for digits 1 to i+1
        cum_ways = list(reduce(lambda acc, x: acc + [acc[-1] + x], digit_ways, [0]))
        # Note: the reduce above is a bit hacky to create a prefix sum. 
        # Correct prefix sum:
        prefix_sum = [sum(digit_ways[:i+1]) for i in range(N)]
        
        # The digit index is the first i where prefix_sum[i] >= rank
        # We use next() with a generator expression.
        d_idx = next(i for i, s in enumerate(prefix_sum) if s >= rank)
        
        # Update rank: subtract ways of all digits smaller than d
        new_rank = rank - (prefix_sum[d_idx-1] if d_idx > 0 else 0)
        new_counts = tuple(counts[i] - (1 if i == d_idx else 0) for i in range(N))
        
        return (new_rank, new_counts, seq + [d_idx + 1])

    # Use reduce to simulate the loop for NK positions
    final_state = reduce(find_digit, range(N * K), (target_rank, initial_counts, []))
    
    # Print the result sequence
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()