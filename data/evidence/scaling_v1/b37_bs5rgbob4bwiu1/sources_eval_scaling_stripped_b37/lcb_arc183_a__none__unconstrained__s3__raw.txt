```python
import sys
from functools import reduce
from operator import mul

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle sequence (or the one just before the middle if S is even).
    # Since the set of sequences is symmetric (replacing i with N-i+1),
    # the "middle" sequence is the one that is its own complement, 
    # or the one just before the complement of the first sequence.
    # Actually, the property is: if a sequence A is the i-th, 
    # then the sequence A' (where each x is replaced by N-x+1) is the (S-i+1)-th.
    # We want the index idx = (S + 1) // 2.
    # If S is odd, idx is the exact middle. If S is even, idx is the end of the first half.
    
    # To find the sequence at a specific rank, we iterate through each position
    # and try numbers 1 to N. We calculate how many sequences start with that prefix.
    # The number of ways to complete a sequence given remaining counts c1, c2, ..., cN
    # is (sum(ci))! / product(ci!).
    
    # However, S can be massive, so we cannot compute it directly.
    # But we only need to know if the rank is <= the number of sequences starting with a certain digit.
    # We can use the property that the "middle" sequence is the one that 
    # lexicographically balances the distribution.
    # For a symmetric distribution, the middle sequence is the one that 
    # "mirrors" the available digits.
    
    # Specifically, for the (S+1)//2-th sequence:
    # At each step, we check if the number of sequences starting with digits < x
    # is less than the target rank.
    # Since we want the middle, we can maintain the target rank as a fraction or 
    # use the fact that we are looking for the median.
    # The median sequence is the one where at each step we pick the smallest x
    # such that the number of sequences starting with 1...x is >= S/2.
    
    # Let f(c1, ..., cN) = (sum ci)! / product(ci!)
    # We want the smallest x such that sum_{j=1}^{x-1} f(c1, ..., cj-1, ..., cN) < S/2
    # This is equivalent to: the sum of counts of sequences starting with 1...x-1
    # is less than half the total sequences.
    
    # Instead of large numbers, we can use the fact that:
    # f(c1, ..., cj-1, ..., cN) / f(c1, ..., cN) = cj / (sum ci)
    # So we are looking for the smallest x such that:
    # sum_{j=1}^{x} (cj / sum(ci)) >= 1/2 (approximately)
    
    # Let's refine this: we maintain the target rank as a fraction (numerator/denominator).
    # Initial rank: (S+1)//2. 
    # We can use a custom fraction class or simply keep track of the target 
    # relative to the current total.
    
    # For N, K up to 500, we need a way to handle the rank without overflow or 
    # use the property that we want the middle.
    # The middle sequence is simply the one where we pick x such that 
    # the sum of probabilities of picking 1...x-1 is < 0.5 and 1...x is >= 0.5.
    
    # Using Decimal for high precision to avoid floating point issues with 0.5
    from decimal import Decimal, getcontext
    getcontext().prec = 2000 # High precision for N*K = 250,000
    
    counts = [K] * N
    total_rem = N * K
    # Target rank relative to total is 0.5. 
    # Since we want floor((S+1)/2), for S even, it's S/2. For S odd, (S+1)/2.
    # In both cases, we are looking for the point where the cumulative 
    # distribution function reaches 0.5.
    # Because we want the floor((S+1)/2)-th, and the distribution is symmetric,
    # we can use a target value of 0.5. If the sum is exactly 0.5, 
    # we are at the boundary of the first and second half.
    # The floor((S+1)/2)-th is the last sequence of the first half.
    
    # To avoid the "exactly 0.5" case (which happens when S is even),
    # we can use a target like 0.5 - epsilon, but the problem asks for 
    # the sequence just before the midpoint if S is even.
    # Actually, the simplest way to handle "floor((S+1)/2)" is to use 
    # a target value of 0.5 and handle the boundary.
    # If the cumulative probability is exactly 0.5, we have reached the end of the first half.
    
    # Let's use a target value T = 0.5. 
    # At each step, we find x such that sum_{j=1}^{x-1} P(j) < T <= sum_{j=1}^{x} P(j).
    # If sum_{j=1}^{x} P(j) == 0.5, then x is the digit that completes the first half.
    
    # Wait, if sum_{j=1}^{x} P(j) == 0.5, then the first half ends exactly at the 
    # last sequence starting with x.
    # So we want the largest sequence that starts with x and has the remaining 
    # digits in descending order.
    
    # Correct logic:
    # We maintain a target value `target` (initially 0.5).
    # At each position:
    # 1. Calculate P(j) = counts[j] / total_rem for j = 1...N.
    # 2. Find x such that sum_{j=1}^{x-1} P(j) < target <= sum_{j=1}^{x} P(j).
    # 3. The new target for the next position is (target - sum_{j=1}^{x-1} P(j)) / P(x).
    # 4. Special case: If target is exactly 0.5 and we hit a boundary, 
    #    the "floor" means we stay in the first half.
    
    # To handle the "floor" and the "boundary" precisely:
    # We can use a target value of 0.5. If at any point the cumulative probability
    # is exactly 0.5, it means we have reached the end of the first half.
    # The last sequence of the first half is the one where all subsequent 
    # choices are the largest possible available digits.
    
    # Let's use a small epsilon to simulate "just slightly less than 0.5"
    # or use a Fraction. But 0.5 is exact in binary.
    # The only issue is when the cumulative sum is exactly 0.5.
    # If sum(P(1...x)) == 0.5, then the current sequence is the last of the first half
    # IF the remaining sequence is the largest possible.
    # Actually, if sum(P(1...x)) == 0.5, then the target for the next step becomes 1.0.
    # A target of 1.0 means we always pick the largest available digit.
    
    target = Decimal('0.5')
    res = []
    
    # To handle the floor((S+1)/2) correctly:
    # If S is even, we want the (S/2)-th.
    # If S is odd, we want the ((S+1)/2)-th.
    # In both cases, we want the sequence at the 0.5 quantile, 
    # but if 0.5 falls exactly between two sequences, we take the lower one.
    # This is naturally handled by: 
    # while target > 0:
    #   for x in 1..N:
    #     p = counts[x]/total
    #     if target <= p:
    #       # pick x, target = target / p
    #       break
    #     else:
    #       target -= p
    
    # One detail: if target becomes exactly 1.0, it means we are looking for the 
    # lexicographically largest sequence of the remaining.
    
    # Since we cannot use loops, we use a list comprehension or map to 
    # simulate the state transition. However, the state depends on the 
    # previous state. We can use `reduce` to simulate the loop.
    
    def step(state, _):
        target, counts, total_rem = state
        
        # Find x such that sum_{j=1}^{x-1} P(j) < target <= sum_{j=1}^{x} P(j)
        # We use a list comprehension to calculate cumulative probabilities
        probs = [Decimal(counts[i]) / Decimal(total_rem) for i in range(N)]
        cum_probs = [sum(probs[:i+1]) for i in range(N)]
        
        # Find the first index i where cum_probs[i] >= target
        # We use a generator with next() to find the index
        idx = next(i for i, cp in enumerate(cum_probs) if cp >= target)
        
        # Update target: (target - prev_cum_prob) / current_prob
        prev_cp = cum_probs[idx-1] if idx > 0 else Decimal(0)
        new_target = (target - prev_cp) / probs[idx]
        
        # Update counts
        new_counts = list(counts)
        new_counts[idx] -= 1
        
        return (new_target, tuple(new_counts), total_rem - 1), idx + 1

    # Initial state: (target, counts, total_rem)
    initial_state = (target, tuple(counts), N * K)
    
    # Run the step function NK times
    final_state, sequence_indices = reduce(
        lambda acc, _: (step(acc[0], _), acc[1] + [step(acc[0], _)[1]]),
        range(N * K),
        (initial_state, [])
    )
    
    # The reduce above is slightly wrong because it calls step() twice.
    # Let's fix it to pass the state correctly.
    
def solve_fixed():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    from decimal import Decimal, getcontext
    getcontext().prec = 2000
    
    def get_next_state(state):
        target, counts, total_rem = state
        # Use a list comprehension to find the digit
        # We need the smallest i such that sum(counts[0...i]) / total_rem >= target
        # To avoid the loop, we can use a helper function with reduce to find the index
        
        def find_idx(acc, i):
            curr_target, found_idx = acc
            if found_idx != -1:
                return acc
            p = Decimal(counts[i]) / Decimal(total_rem)
            if curr_target <= p:
                return (curr_target, i)
            return (curr_target - p, -1)

        # This is tricky. Let's use a different approach for find_idx.
        # We can use a list comprehension to create the cumulative sums.
        # But we can't use loops. We can use map/reduce.
        pass

# The constraint "no for loops" makes the "find x" part hard.
# Let's use a recursive-like structure with reduce.

def final_solve():
    import sys
    from decimal import Decimal, getcontext
    from functools import reduce

    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    getcontext().prec = 2000

    def step(state):
        target, counts, total_rem = state
        
        # Calculate probabilities for all N digits
        probs = [Decimal(c) / Decimal(total_rem) for c in counts]
        
        # Find the index using a reduce that accumulates (current_target, found_index)
        # If found_