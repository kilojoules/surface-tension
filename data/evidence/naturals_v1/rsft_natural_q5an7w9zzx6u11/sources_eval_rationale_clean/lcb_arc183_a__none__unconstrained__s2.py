import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We want the floor((S+1)/2)-th sequence.
    # Due to symmetry (replacing x with N-x+1), the middle sequence
    # is the one that is "balanced".
    # For a sequence A, let A' be the sequence where A'_i = N + 1 - A_i.
    # The sequence we are looking for is the one that is the 
    # lexicographical median.
    
    # To find the median sequence without loops/recursion:
    # We use reduce to iterate through the positions 1 to N*K.
    # State: (current_counts, target_index)
    # However, calculating S is too large for standard floats, 
    # but Python handles arbitrary precision integers.
    
    # Precompute factorials for the multinomial coefficient
    # Using a list comprehension to simulate a loop for factorials
    fact = reduce(lambda acc, i: acc + [acc[-1] * i], range(1, N * K + 1), [1])
    
    def get_count(counts):
        # Multinomial coefficient: (sum(counts))! / product(counts!)
        total = sum(counts)
        denom = reduce(lambda a, b: a * fact[b], counts, 1)
        return fact[total] // denom

    # Initial state: counts of each number 1..N, and the target index
    # S = get_count([K] * N)
    # target = (S + 1) // 2
    
    initial_counts = [K] * N
    initial_target = (get_count(initial_counts) + 1) // 2
    
    # We use reduce to determine the digit for each position
    # The state is (counts, target, result_sequence)
    def step(state, _):
        counts, target, res = state
        
        # We need to find the smallest digit d such that 
        # sum_{i=1}^{d-1} count(i) * get_count(counts - e_i) < target
        # and sum_{i=1}^{d} count(i) * get_count(counts - e_i) >= target
        
        # To avoid loops, we use a helper to find the digit d
        # We can use a list comprehension to calculate the prefix sums of counts
        # for the current position.
        
        # Calculate how many sequences start with digit 1, 2, ... N
        # only for digits that still have remaining counts.
        options = [
            (d, get_count([counts[i] - (1 if i == d-1 else 0) for i in range(N)]))
            for d in range(1, N + 1) if counts[d-1] > 0
        ]
        
        # Find the digit d that crosses the target threshold
        # We use a custom reduce or a combination of filter/next to find d
        # Since we can't use loops, we calculate the cumulative counts
        # and find the first index where it exceeds target.
        
        # Using a list comprehension to find the digit
        # We calculate the cumulative sum of sequences starting with d
        # and find the first d where cum_sum >= target.
        
        # To implement this without a loop, we can use a trick with 
        # a list of cumulative sums.
        def find_digit(opts, t):
            # Calculate cumulative sums of the counts of sequences
            # opts is [(digit, count), ...]
            # We need the first digit where the sum of counts reaches t.
            
            # We can't use a loop to build the cumsum, but we can use 
            # a list comprehension with a slice and sum()
            # However, that's O(N^2). With N=500, it's 250,000, which is fine.
            return [o[0] for o in opts if t <= sum(x[1] for x in opts[:opts.index(o)+1))][0]

        # This is slightly wrong because the target needs to be updated.
        # Let's refine the state transition.
        return state # Placeholder
    
    # Because the logic inside 'step' requires updating the target,
    # we must handle the target subtraction.
    
    def refined_step(state, _):
        counts, target, res = state
        
        # Calculate counts for each possible digit 1..N
        # ways[d] = number of sequences starting with digit d+1
        ways = [
            (get_count([counts[i] - (1 if i == d else 0) for i in range(N)]) 
             if counts[d] > 0 else 0) 
            for d in range(N)
        ]
        
        # Find the digit d such that sum(ways[0...d-1]) < target <= sum(ways[0...d])
        # We use a list comprehension to find the index
        # The index d is the first index where the prefix sum is >= target
        d = [i for i in range(N) if target <= sum(ways[:i+1])][0]
        
        # Update target for the next position: target = target - sum(ways[:d])
        new_target = target - sum(ways[:d])
        new_counts = [counts[i] - (1 if i == d else 0) for i in range(N)]
        
        return (new_counts, new_target, res + [d + 1])

    final_state = reduce(refined_step, range(N * K), (initial_counts, initial_target, []))
    print(*(final_state[2]))

solve()