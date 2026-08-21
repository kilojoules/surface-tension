The constraint to avoid explicit loops forces the use of functional programming paradigms, which can lead to more concise, declarative code by focusing on "what" to compute rather than "how" to iterate. To solve this problem, I will use `functools.reduce` to maintain the state of remaining counts of each number and the current index, and list comprehensions for mapping and filtering. Since we need the median sequence, and the total number of sequences is symmetric, the target sequence is the one that "balances" the distribution. For a given position, we can determine the digit by calculating how many sequences start with digits smaller than $d$ and comparing it to the target rank. However, since the total number of sequences $S$ can be massive, we use the property that the median sequence is the one where we try to pick the smallest possible digit that keeps the remaining permutations $\ge$ half of the total remaining permutations.

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

    # The total number of good sequences is S = (NK)! / (K!^N)
    # We need the floor((S+1)/2)-th sequence.
    # This is the middle sequence. Due to the symmetry of the set of 
    # good sequences (replacing x with N-x+1), the median sequence
    # is the one that is "self-dual" or the first one of the second half.
    # Specifically, for any sequence A, its dual A' is also a good sequence.
    # A < A' if at the first index i where they differ, A_i < A'_i.
    # The median sequence is the one that is lexicographically the 
    # "middle" one. A key property is that the median sequence is 
    # the one that mirrors the distribution.
    
    # For this specific problem, the target index is (S+1)//2.
    # We can determine the elements one by one.
    # At each step, we check if the number of sequences starting with 
    # digits 1...d is less than the target rank.
    
    # However, calculating S is impossible with large N, K.
    # But we only need to compare the rank with S/2.
    # The number of sequences starting with digit d is:
    # (Total_Remaining - 1)! / (K_1! ... (K_d-1)! ... K_N!)
    # We want the smallest d such that:
    # sum_{i=1}^{d} (ways starting with i) >= (Total_Ways + 1) / 2
    
    # Using the property of symmetry: the median sequence is the one
    # where we pick d such that the number of sequences starting with 
    # 1...d-1 is < S/2 and 1...d is >= S/2.
    # This is equivalent to saying we pick d such that the number of 
    # sequences starting with 1...d-1 is less than the number of 
    # sequences starting with d+1...N.
    
    # Let f(counts) be the number of ways to arrange the remaining digits.
    # f(c1, ..., cn) = (sum ci)! / prod(ci!)
    # We compare sum_{i=1}^{d-1} f(counts with i decremented) vs 
    # sum_{i=d+1}^{n} f(counts with i decremented).
    
    # Since we cannot use loops, we use reduce to iterate through the 
    # NK positions.
    
    def get_digit(state):
        counts, target_half = state
        # We seek the smallest d such that:
        # sum_{i=1}^{d} ways(i) >= (Total_Ways + 1) / 2
        # This is equivalent to:
        # sum_{i=1}^{d-1} ways(i) < sum_{i=d+1}^{n} ways(i) + ways(d)/2
        
        # To avoid large numbers and loops, we can observe that 
        # the median sequence is simply the one where we pick d
        # such that the number of ways to complete the sequence 
        # using digits < d is less than or equal to the number of 
        # ways using digits > d.
        
        # Actually, a simpler observation: the median sequence is 
        # the one that is "balanced". For the first position, 
        # if N=3, K=3, the digits are 1, 2, 3.
        # Ways starting with 1: (8!)/(2!3!3!) = 560
        # Ways starting with 2: (8!)/(3!2!3!) = 560
        # Ways starting with 3: (8!)/(3!3!2!) = 560
        # Total S = 1680. Target = 840.
        # 1s: 1-560, 2s: 561-1120, 3s: 1121-1680.
        # Target 840 falls in the '2' range.
        
        # To find d without loops:
        # We need d such that sum_{i=1}^{d-1} ways(i) < S/2 <= sum_{i=1}^{d} ways(i)
        # ways(i) = (Total-1)! / (K_i-1)! prod_{j!=i} (K_j!)
        # ways(i) / Total_Ways = K_i / Total
        
        # So we need sum_{i=1}^{d-1} (K_i / Total) < 1/2 <= sum_{i=1}^{d} (K_i / Total)
        # sum_{i=1}^{d-1} K_i < Total/2 <= sum_{i=1}^{d} K_i
        
        # This is only true for the first digit. For subsequent digits, 
        # the "remaining" S changes. But the symmetry holds!
        # The target rank is always the median of the current remaining set.
        
        # Let's use a helper to find d:
        def find_d(current_counts):
            total_rem = sum(current_counts)
            # We need smallest d such that sum_{i=1}^d counts[i-1] >= (total_rem + 1) / 2
            # We can use a list comprehension to find all prefix sums and then filter.
            prefix_sums = reduce(lambda acc, x: acc + [acc[-1] + x], current_counts, [0])
            # prefix_sums[d] is sum of first d counts.
            # We want smallest d such that prefix_sums[d] >= (total_rem + 1) / 2
            # Since we can't loop, we use a list comprehension to find indices.
            indices = [d for d in range(1, n + 1) if prefix_sums[d] >= (total_rem + 1) / 2]
            return indices[0]

        d = find_d(counts)
        new_counts = list(counts)
        new_counts[d-1] -= 1
        return (new_counts, d)

    # Initial state: (counts, first_digit)
    # We run this NK times.
    initial_counts = [k] * n
    final_state = reduce(
        lambda state, _: (
            # The state is (current_counts, sequence_so_far)
            (lambda res: (res[0], state[1] + [res[1]])) (get_digit(state[0]))
        ) if 0 else None, # This structure is just to allow the helper
        range(n * k),
        ([k] * n, [])
    )
    
    # The above reduce is slightly wrong because get_digit needs the current counts.
    # Let's redefine the reduce logic.
    
    def run_process(n, k):
        def step(state, _):
            counts, seq = state
            # Find d such that sum_{i=1}^{d-1} counts[i-1] < (sum(counts)+1)/2 <= sum_{i=1}^d counts[i-1]
            total_rem = sum(counts)
            # Use a list comprehension to find the first d that satisfies the condition
            # We can't use a loop, so we create a list of booleans and find the first True index.
            # prefix_sums = [sum(counts[:i]) for i in range(n + 1)]
            # d = next(i for i in range(1, n + 1) if prefix_sums[i] >= (total_rem + 1) / 2)
            
            # To avoid 'next' and 'for' loops, we use a list comprehension and index 0.
            # We use the property that we need the smallest d.
            d = [i for i in range(1, n + 1) if sum(counts[:i]) >= (total_rem + 1) / 2][0]
            
            new_counts = [counts[i] - (1 if i == d-1 else 0) for i in range(n)]
            return (new_counts, seq + [d])

        result = reduce(step, range(n * k), ([k] * n, []))
        return result[1]

    # Since the prompt forbids loops and recursion, and requires a functional style:
    # The logic inside 'step' uses list comprehensions and sum(), which are allowed.
    # The 'reduce' handles the iteration.
    
    # Wait, the logic 'sum(counts[:i]) >= (total_rem + 1) / 2' is for the median of 
    # the current distribution. Let's verify with Sample 1: N=2, K=2.
    # Total = 4. Target = (6+1)//2 = 3.
    # Pos 1: counts=[2,2], total=4. d=1: sum([2])=2. 2 >= 2.5? No. d=2: sum([2,2])=4. 4 >= 2.5? Yes. d=2.
    # Seq=[2], counts=[2,1].
    # Pos 2: counts=[2,1], total=3. Target rank in remaining is tricky.
    # The rank shifts. Let's use the property: 
    # The median sequence is the one that is its own "complement" (reversed and values flipped).
    # For N=2, K=2, sequences are: 1122, 1212, 1221, 2112, 2121, 2211.
    # Median is 3rd: 1221.
    # My logic: Pos 1: counts=[2,2], total=4. Target rank 3.
    # Ways starting with 1: 3!/(2!1!) = 3.
    # Since 3 >= 3, the first digit is 1.
    # Remaining: counts=[1,2], target rank 3 - 0 = 3. (Wait, if we pick 1, the rank is relative to the block of 1s).
    # If we pick digit d, the new rank is: rank - sum_{i=1}^{d-1} ways(i).
    
    # Correct logic with rank:
    def solve_with_rank(n, k):
        import math
        def nCr_fact(n, ks):
            # This is slow, but we can't use loops. 
            # However, we only need to compare.
            # ways(d) = (sum(counts)-1)! / prod(counts_i!) * counts[d-1]
            pass

        # Given the constraints and the "no loop" rule, the most reliable way 
        # to find the median is to realize that the median sequence is the 
        # lexicographical middle. For a symmetric distribution, the 
        # middle sequence is the one that is "self-complementary" 
        # in terms of the available digits.
        # The digit d at position p is the one that splits the remaining 
        # permutations into two equal halves.
        # This happens when the sum of counts of digits < d is 
        # roughly equal to the sum of counts of digits > d.
        
        # For the median, we want the smallest d such that:
        # count(1...d-1) < (Total+1)/2 <= count(1...d)
        # This is exactly what I wrote! Let's re-check Sample 1:
        # N=2, K=2. Total S=6. Target=3.
        # Pos 1: counts=[2,2], total=4. 
        # ways(1) = 3!/(1!2!) = 3.
        # Rank 3 <= ways(1)? Yes. So d=