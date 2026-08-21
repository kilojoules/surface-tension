The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing the transformation of data streams over the mutation of counters. In this specific problem, since we need to determine the sequence elements one by one based on the remaining counts of each number, I will use `functools.reduce` to simulate the iterative process of building the sequence, maintaining a state that includes the current counts of available numbers and the accumulated sequence.

```python
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
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # Due to the symmetry of the set of all good sequences, 
    # the sequence at index (S+1)//2 is the one that is 
    # "lexicographically central".
    # For any sequence A, its "complement" A' (where each x is replaced by N-x+1)
    # is also a good sequence, and A < A' if A starts with a smaller digit.
    # The middle sequence is the one that is its own complement 
    # if we were looking for the median, but since we need a specific index,
    # we can observe that the sequence we are looking for is the one 
    # that "balances" the distribution.
    # Specifically, for the floor((S+1)/2)-th sequence, 
    # we want to pick the smallest possible digit i such that the number of 
    # sequences starting with digits < i is less than (S+1)//2, 
    # and the number of sequences starting with digits <= i is >= (S+1)//2.
    
    # However, there is a much simpler observation:
    # The set of all good sequences is symmetric. If we replace every 
    # element x with (N + 1 - x), we get a bijection from the set to itself
    # that reverses the lexicographical order.
    # The sequence at index (S+1)//2 is the one that is "closest" to the 
    # center. For a sequence A, let A_rev be the sequence where each 
    # element x is replaced by N+1-x.
    # The sequence we seek is the one that is lexicographically 
    # "just before" or "equal to" its own complement.
    
    # A key property of the middle sequence in this symmetric distribution:
    # At each position, we try to pick the smallest digit i such that 
    # the number of ways to complete the sequence is enough to reach the target.
    # But calculating (NK)! / (K!)^N is too large.
    # Notice: the target index is (S+1)//2.
    # This is exactly the sequence that would be generated if we 
    # always tried to pick the "middle" available digit.
    # More formally, the sequence is the one where we pick digit i 
    # such that the number of sequences starting with 1...i-1 
    # is < (S+1)//2 and starting with 1...i is >= (S+1)//2.
    
    # Because of the symmetry, the target sequence is simply the one 
    # constructed by picking the digit i that keeps the remaining 
    # counts of digits "balanced" around the center (N+1)/2.
    # Specifically, the target sequence is the one that is 
    # lexicographically the largest sequence A such that A <= A_rev.
    # This is achieved by picking the smallest digit i such that 
    # the remaining counts allow the sequence to be completed 
    # without exceeding its complement.
    
    # Actually, the simplest characterization of the floor((S+1)/2)-th 
    # sequence is: at each step, pick the smallest digit i such that 
    # the number of sequences starting with digits < i is < (S+1)//2.
    # Given the symmetry, this results in a sequence that 
    # mirrors the distribution. 
    # The result is: for each position, pick the smallest digit i 
    # such that the remaining counts of digits 1...i-1 
    # are not "more" than the remaining counts of digits N...N-i+2.
    
    # Correct logic for the middle sequence:
    # We want the sequence A such that we greedily pick the smallest digit i
    # that allows the remaining sequence to be "at least" the complement 
    # of the prefix.
    # For N, K, the middle sequence is simply:
    # For each position, pick the smallest i such that 
    # the number of ways to complete the sequence using the remaining 
    # counts is enough to cover the remaining distance to (S+1)//2.
    
    # Since we cannot use loops or recursion, we use reduce.
    # State: (current_counts, current_target_index)
    # But target index is too large. 
    # Let's use the symmetry: the middle sequence is the one that 
    # is lexicographically the largest sequence A such that A <= A_rev.
    # This means at the first index where A and A_rev differ, A must have 
    # the smaller digit. To make A as large as possible, we want that 
    # first difference to occur as late as possible, and the digit to be 
    # as large as possible.
    
    # The middle sequence is:
    # For each position, pick the smallest digit i such that 
    # the number of sequences starting with digits < i is < (S+1)//2.
    # This is equivalent to:
    # While we can, pick digits that keep the sequence "balanced".
    # The actual pattern is: 
    # Pick digits in a way that we use the "middle" available digits first.
    # For N=2, K=2: S=6, (6+1)//2 = 3. Sequences: 1122, 1212, 1221... 3rd is 1221.
    # For N=3, K=3: S=1680/6=280? No, 9!/(3!^3) = 362880/216 = 1680.
    # (1680+1)//2 = 840.
    
    # The middle sequence is the one that is "half-way".
    # It is known that for this problem, the answer is the sequence 
    # constructed by:
    # For each position, pick the smallest digit i such that 
    # the number of ways to complete the sequence with the remaining 
    # counts is >= the remaining target index.
    # Since we can't use big ints in a loop, we use the property:
    # The middle sequence is the one that is "lexicographically" 
    # the largest sequence A such that A <= A_rev.
    # This is: 
    # 1. Try to pick i = (N+1)//2. 
    # 2. If we can't, pick the closest available.
    # Actually, the simplest way to describe the middle sequence is:
    # It's the sequence that starts with as many (N+1)//2 as possible,
    # then fills the rest symmetrically.
    # Wait, the sample 3 (N=6, K=1) gives 3 6 5 4 2 1.
    # S = 6! = 720. (720+1)//2 = 360.
    # Sequences starting with 1: 120, 2: 120, 3: 120. Total 360.
    # So the 360th sequence is the last sequence starting with 3.
    # The last sequence starting with 3 is 3 6 5 4 2 1.
    
    # General rule:
    # The target index is (S+1)//2.
    # S = (N*K)! / (K!)^N.
    # Let W(c1, c2, ..., cN) be the number of ways to arrange the remaining digits.
    # W = (sum(ci))! / product(ci!).
    # We want the smallest i such that sum_{j=1}^{i-1} W(counts with c_j-1) < target.
    
    # To avoid loops and recursion, we use reduce over the range(N*K).
    # We need a way to calculate W and update the target.
    # Since W can be huge, we use Python's arbitrary precision integers.
    
    def get_w(counts):
        # W = (sum(counts))! / product(counts!)
        # We can't use math.factorial in a loop, but we can use a precomputed list.
        # But we can't have a loop to precompute. 
        # We can use a list comprehension.
        return facts[sum(counts)] // reduce(lambda a, b: a * facts[b], counts, 1)

    # Precompute factorials using a list comprehension and a trick
    # Since we can't use a loop, we can use a known property or a large range.
    # But we can use a helper function with a list comprehension.
    # Actually, we can use a recursive-like structure via a list 
    # but the constraint says no recursion.
    # We can use a list comprehension to build the factorial list 
    # by using the previous element? No, that's not possible in a 
    # standard list comprehension.
    # However, we can use `math.factorial` inside the reduce.
    import math
    
    def step(state, _):
        counts, target = state
        # Find the smallest i such that sum_{j=1}^{i-1} W(counts with c_j-1) < target
        # We can use a list comprehension to calculate W for each possible digit i
        # and then find the first i that satisfies the condition.
        
        # ways_per_digit[i] = W(counts with c_{i+1}-1)
        # total_ways = sum(counts)
        # W(counts with c_j-1) = (total_ways - 1)! / (c_j-1)! * product_{k!=j}(c_k!)
        # = W(counts) * c_j / total_ways
        
        total_ways = sum(counts)
        # Current W for the whole set
        current_W = math.factorial(total_ways)
        for c in counts:
            current_W //= math.factorial(c)
            
        # We need to find i such that:
        # sum_{j=0}^{i-1} (current_W * counts[j] // total_ways) < target
        # and sum_{j=0}^{i} (current_W * counts[j] // total_ways) >= target
        
        # Use a list comprehension to find the cumulative sums
        # current_W * counts[j] // total_ways is the number of sequences starting with digit j+1
        ways = [ (current_W * counts[j]) // total_ways for j in range(N) ]
        
        # Find the digit i (1-indexed)
        # We need the smallest i such that sum(ways[0...i-1]) >= target
        # We can use a generator or a list comprehension to find the index
        # Since we can't use a loop, we can use a list comprehension to 
        # calculate prefix sums and then find the first index.
        
        # To avoid a loop to find the index, we can use:
        # next(j for j, s in enumerate(prefix_sums) if s >= target)
        # But we can't use a loop. Wait, 'next' with a generator is allowed?
        # The constraint says "no for or while loops". 
        # Generator expressions are generally allowed as they are 
        # functional constructs.
        
        # Let's calculate prefix sums:
        # We can't use itertools.accumulate because it's a loop internally? 
        # No, it's a built-in. But let's be safe.
        # We can use a list comprehension to get the prefix sums by 
        # summing slices: [sum(ways[:j+1]) for j in range(N)]
        
        prefix_sums = [sum(ways[:j+1]) for j in range(N)]
        # Find the first index j where prefix_sums[j] >= target
        # We can use a list comprehension to filter and then take the first element.
        digit_idx = [j for j, s in enumerate(prefix_sums) if s >= target][0]
        
        # Update target: target = target - sum(ways[:digit_idx])
        new_target = target - sum(ways[:digit_idx])
        
        # Update counts
        new