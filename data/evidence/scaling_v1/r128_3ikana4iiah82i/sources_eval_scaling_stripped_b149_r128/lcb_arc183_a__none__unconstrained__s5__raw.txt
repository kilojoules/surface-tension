```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. 
    # Due to the symmetry of the set of all good sequences, 
    # if we replace every element x in a sequence with (N + 1 - x),
    # we get another good sequence.
    # This operation is a bijection that reverses lexicographical order.
    # Therefore, the floor((S+1)/2)-th sequence is the one that is 
    # "lexicographically central".
    
    # For a fixed N and K, the sequence that is the floor((S+1)/2)-th 
    # is the one where we try to place the smallest available numbers 
    # as late as possible and largest as early as possible, 
    # but balanced across the whole string.
    
    # Actually, the property of the "middle" sequence in this specific 
    # combinatorial problem is that it is the lexicographical 
    # complement of itself if S is odd, or the one just before the 
    # complement if S is even.
    
    # A known construction for the middle sequence of this problem:
    # The sequence is formed by blocks. 
    # For N=3, K=3, the answer is 2 2 2 1 3 3 3 1 1.
    # This looks like: 
    # Block of K copies of (N+1)//2, 
    # then Block of K copies of 1, 
    # then Block of K copies of N, 
    # then Block of K copies of 2, 
    # then Block of K copies of N-1...
    
    # Let's refine the pattern:
    # The middle sequence starts with the middle value (if N is odd).
    # Then it alternates between the smallest and largest remaining values.
    # If N is even, it starts with the smaller of the two middle values.
    
    # Correct logic for the middle sequence:
    # We want the sequence that is as close to the "average" as possible.
    # The pattern is:
    # While numbers remain:
    # 1. Pick the middle-most available number.
    # 2. Pick the smallest available number.
    # 3. Pick the largest available number.
    # Repeat.
    
    # However, the sample 3 (6 1) -> 3 6 5 4 2 1 suggests a different pattern.
    # Let's analyze Sample 3: N=6, K=1. Middle is 3rd or 4th.
    # Sequences: (1,2,3,4,5,6) ... (6,5,4,3,2,1)
    # The 3rd sequence of 6! is (1,2,3,4,6,5) - wait, the sample says 3 6 5 4 2 1.
    # Let's re-read: Sample 3: 6 1 -> 3 6 5 4 2 1.
    # This is the 360th sequence (6!/2).
    # For N=6, K=1, S=720. floor(721/2) = 360.
    # The 360th permutation of (1,2,3,4,5,6) is indeed (3, 6, 5, 4, 2, 1).
    
    # The general algorithm to find the m-th permutation of a multiset:
    # For each position, try digits d = 1...N.
    # Calculate how many sequences start with the current prefix + d.
    # If m is greater than this count, subtract count from m and try d+1.
    # Otherwise, the digit at this position is d.
    
    # Since we need the middle one, we can use the fact that the 
    # "middle" sequence is the one that starts with the digit 
    # that splits the total count S into two halves.
    
    # For a multiset, the number of permutations is (sum(k_i))! / product(k_i!).
    # We can use a helper to calculate this.
    
    import math
    
    def count_permutations(counts):
        total = sum(counts)
        # Using math.prod and math.factorial
        # S = total! / (k1! * k2! ... )
        denom = math.prod(math.factorial(c) for c in counts)
        return math.factorial(total) // denom

    # We need to find the sequence for m = (S + 1) // 2
    # Instead of a loop, we use reduce to build the sequence.
    # State: (current_counts, current_m)
    
    initial_counts = [k] * n
    total_s = count_permutations(initial_counts)
    initial_m = (total_s + 1) // 2
    
    def step(state, _):
        counts, m = state
        # Try each digit d from 1 to N
        # We need to find the first d such that the sum of permutations 
        # of prefixes starting with 1...d-1 is < m, 
        # and the sum of permutations starting with 1...d is >= m.
        
        # To avoid a loop over N, we can use a list comprehension and 
        # a custom search, but since N is 500, a loop inside reduce 
        # is the only way. The constraint says "no loops", but 
        # "reduce" is allowed. We can use a generator expression 
        # inside 'next()' to find the digit.
        
        # For a digit d, the number of ways to complete the sequence is:
        # count_permutations(counts with counts[d-1] decremented)
        
        # We find the digit d by iterating 1...N
        # We use a generator to find the first d that satisfies the condition.
        # Since we can't use a loop, we'll use a trick with 'next' 
        # and a running total of permutations.
        
        # However, we need the running total of permutations to compare with m.
        # We can use another reduce or a mathematical approach to find the digit.
        
        # Let's use a list comprehension to calculate the counts for all d,
        # then use a technique to find the index.
        
        # Calculate permutations for each possible next digit
        # ways[d] = count_permutations(counts - e_d)
        # We need the smallest d such that sum(ways[1...d]) >= m
        
        # To do this without a loop:
        # 1. Create a list of ways for all d where counts[d] > 0
        # 2. Use a mathematical approach to find the index.
        
        # Since we can't use loops, we use a list comprehension to 
        # build the 'ways' list and then use a logic to find the digit.
        
        # But wait, the 'state' needs to be updated.
        # The digit d is the one where:
        # sum(ways[i] for i in range(d-1)) < m <= sum(ways[i] for i in range(d))
        
        # We can find d using a list comprehension and 'next':
        # d = next(i for i in range(1, n+1) if ...)
        # But we need the prefix sum.
        
        # Let's use a more functional approach.
        # We can pre-calculate the ways for all d.
        # Then we find the index by filtering the list.
        
        # Actually, the most efficient way to find the digit d is to 
        # iterate through the digits and keep track of the cumulative sum.
        # Since we can't use a loop, we can use a nested reduce or 
        # a clever list comprehension.
        
        # Let's use a list comprehension to find the first index where 
        # the cumulative sum exceeds m.
        
        # To get the cumulative sum without a loop, we can use 
        # itertools.accumulate.
        
        from itertools import accumulate
        
        # Calculate ways for each digit 1...N
        # Only if counts[i] > 0
        ways = [
            count_permutations([counts[j] - (1 if j == i else 0) for j in range(n)])
            if counts[i] > 0 else 0
            for i in range(n)
        ]
        
        # Cumulative sum of ways
        cum_sum = list(accumulate(ways))
        
        # Find the first index i where cum_sum[i] >= m
        # We use next() with a generator expression
        digit_idx = next(i for i, s in enumerate(cum_sum) if s >= m)
        
        # Update m for the next position:
        # The new m is the original m minus the sum of ways of all digits < digit_idx
        prev_sum = cum_sum[digit_idx - 1] if digit_idx > 0 else 0
        new_m = m - prev_sum
        
        # Update counts
        new_counts = list(counts)
        new_counts[digit_idx] -= 1
        
        return (new_counts, new_m), digit_idx + 1

    # We need to run the 'step' function NK times.
    # We use reduce to iterate NK times and collect the digits.
    # The state in reduce will be ((counts, m), result_list)
    
    final_state = reduce(
        lambda acc, _: (step(acc[0], _), acc[1] + [step(acc[0], _)[1]]),
        range(n * k),
        ((initial_counts, initial_m), [])
    )
    
    # The above reduce is slightly wrong because it calls step() twice.
    # Let's fix it to call step once.
    
    def fixed_reduce(acc, _):
        state, res = acc
        new_state, digit = step(state, _)
        return (new_state, res + [digit])

    # Re-running the logic inside a clean structure:
    # Since I cannot use a loop, I will define the logic inside a 
    # single reduce call.
    
    # To avoid the double call to step(), I'll use a helper 
    # that returns the updated state and the digit.
    
    # Let's redefine the whole process.
    pass

# Because the constraints forbid loops and require a specific format,
# I will implement the logic using a single reduce and a helper function.

def solve_final():
    import sys
    import math
    from itertools import accumulate
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data: return
    n, k = map(int, input_data)

    def count_perms(counts):
        return math.factorial(sum(counts)) // math.prod(map(math.factorial, counts))

    def get_next_digit(state):
        counts, m = state
        # Calculate ways for each digit 1..n
        ways = [
            count_perms([counts[j] - (1 if j == i else 0) for j in range(n)])
            if counts[i] > 0 else 0
            for i in range(n)
        ]
        cum_sum = list(accumulate(ways))
        digit_idx = next(i for i, s in enumerate(cum_sum) if s >= m)
        
        # Update state
        new_counts = list(counts)
        new_counts[digit_idx] -= 1
        prev_sum = cum_sum[digit_idx - 1] if digit_idx > 0 else 0
        return (new_counts, m - prev_sum), digit_idx + 1

    # Initial state
    s_total = count_perms([k] * n)
    initial_state = ([k] * n, (s_total + 1) // 2)

    # Use reduce to generate the sequence
    # We store (current_state, sequence_list)
    result = reduce(
        lambda acc, _: (get_next_digit(acc[0])[0], acc[1] + [get_next_digit(acc[0])[1]]),
        range(n * k),
        (initial_state, [])
    )
    
    # Wait, the lambda still calls get_next_digit twice. 
    # To fix this without a loop or assignment, we can use a 
    # nested reduce or a trick with a list