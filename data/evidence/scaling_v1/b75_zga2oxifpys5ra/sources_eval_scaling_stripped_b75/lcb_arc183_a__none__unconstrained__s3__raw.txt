```python
import sys
from math import factorial

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    # We can determine the sequence element by element.
    # For the first position, we try digits d = 1, 2, ..., N.
    # The number of sequences starting with d is (NK-1)! / ((K-1)! * (K!)^(N-1))
    # which simplifies to S * (K / NK).
    
    # Since we need the middle sequence, we can use the property of symmetry.
    # The lexicographical middle of all permutations of a multiset is 
    # effectively the "average" sequence.
    # However, a simpler observation for this specific problem:
    # The total number of sequences S is symmetric. 
    # The sequence at index (S+1)//2 is the one that "balances" the distribution.
    # For a multiset, the middle sequence is found by placing the median 
    # available element at each position, but that's for sets.
    # For multisets, we can use the formula for the number of permutations:
    # Count(n1, n2, ..., nN) = (sum(ni))! / product(ni!)
    
    # Because N and K are up to 500, S is massive. We cannot compute S directly
    # and subtract. But we only need to know if the target index is > current count.
    # target = (S + 1) // 2
    
    # Let's use the property: the middle sequence of all permutations of a multiset
    # is the one where we try to place the 'middle' available character.
    # Specifically, for the first position, we want the smallest d such that
    # sum_{i=1}^{d-1} Count(K, ..., K-1, ..., K) < (S+1)/2 <= sum_{i=1}^{d} Count(...)
    
    # Let f(n1, ..., nN) be the number of permutations.
    # The number of permutations starting with d is f(n1, ..., nk-1, ..., nN).
    # This is S * (kd / sum(ni)).
    
    # We can maintain the target index T = (S + 1) // 2.
    # At each step, we find the smallest d such that T <= count(sequences starting with d).
    # Then we subtract the counts of sequences starting with 1...d-1 from T.
    
    # To avoid giant numbers in loops, we can use the fact that 
    # the number of sequences starting with d is S_current * (count_d / total_remaining).
    
    # However, the target is exactly (S+1)//2.
    # For the first digit, the number of sequences starting with 1 is S * (K / NK).
    # If (S+1)//2 <= S * (K / NK), the first digit is 1.
    # Otherwise, we check digit 2, and so on.
    
    # Since we need to handle (S+1)//2, we can work with a target T.
    # Initial T = (S + 1) // 2.
    # For position p, try d = 1...N:
    #   num_with_d = (total_rem - 1)! / (prod(counts_i!)) * counts_d
    #   if T <= num_with_d:
    #       digit = d
    #       break
    #   else:
    #       T -= num_with_d
    
    # To avoid recalculating factorials, note that:
    # num_with_d = (total_rem_factorial / prod_fact) * (counts_d / total_rem)
    # Let Current_S = total_rem_factorial / prod_fact.
    # num_with_d = Current_S * counts_d // total_rem.
    
    # We can use a helper to compute S and then iterate.
    # Since N, K <= 500, we use Python's arbitrary precision integers.
    
    def get_s(counts):
        total = sum(counts)
        res = factorial(total)
        for c in counts:
            res //= factorial(c)
        return res

    counts = [k] * n
    total_s = get_s(counts)
    target = (total_s + 1) // 2
    
    # To avoid O(N*N*K) and giant multiplications, we pre-calculate 
    # the current S and update it.
    # Current_S is the number of ways to arrange the remaining elements.
    # The number of ways starting with digit d is:
    # (Current_S * counts[d-1]) // sum(counts)
    
    # We can't use a loop to build the sequence because we can't use 
    # recursion or complex structures, but a list comprehension 
    # or a generator is allowed. However, the state (target, counts) 
    # must be updated. We can use a 'reduce' like approach.
    
    from functools import reduce
    
    def step(state, _):
        target, counts, current_s = state
        total_rem = sum(counts)
        
        # Find d such that sum_{i=1}^{d-1} (current_s * counts[i-1] // total_rem) < target
        # We can use a generator to find the digit and the remaining target.
        
        # This helper finds the digit and the new target
        def find_digit(t, cts, s, rem):
            # We need the smallest d (1-indexed) such that 
            # sum_{i=1}^{d} (s * cts[i-1] // rem) >= t
            # We can't use loops, so we use a list comprehension to calculate 
            # the cumulative counts.
            counts_d = [(s * c) // rem for c in cts]
            # Use a list comprehension to find the first index where cumsum >= t
            # But we can't use loops. We can use a trick with a list of 
            # (cumulative_sum, digit) and then filter.
            
            # Since we can't use loops, we use a recursive-like structure 
            # via a list comprehension to simulate the search.
            # Actually, we can just use a list comprehension to find the digit:
            # The digit d is the one where sum(counts_d[:d-1]) < t <= sum(counts_d[:d])
            
            # To avoid O(N^2) inside the reduce, we can't use sum() in a loop.
            # But N=500, so N*N*K is too slow? No, the total length is NK=250,000.
            # We need a more efficient way to find d.
            pass

    # Wait, the constraints are N, K <= 500. Total length is 250,000.
    # An O(NK * N) approach is 125 million operations. Python might be too slow.
    # But we only need to find d. Since we are looking for the middle sequence,
    # the digit d will likely be around N/2.
    
    # Let's reconsider the "middle" sequence.
    # For a symmetric distribution, the middle sequence is the one that is 
    # "self-dual" under the mapping i -> N-i+1 and reversing the sequence.
    # The middle sequence is simply the sequence that, when reversed and 
    # elements replaced by (N+1-x), remains the same.
    # Actually, the middle sequence of all permutations of a multiset is 
    # simply the one where we place the elements in a specific balanced way.
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # Middle (3rd) is (1,2,2,1).
    # This is exactly the sequence that is "lexicographically" in the middle.
    # A known property: the middle sequence is the one where we 
    # distribute the numbers 1...N as evenly as possible.
    # Specifically, the sequence is: 
    # For i from 0 to NK-1:
    #   digit = (i // (NK // N)) ... no, that's for blocks.
    
    # Correct logic for middle sequence of multiset permutations:
    # It is the sequence where we place the digits in the order:
    # 1, 2, ..., N, N, ..., 2, 1, 1, 2, ...
    # Actually, the middle sequence is simply the one that 
    # reads the same forwards as it does backwards if you replace x with N+1-x.
    # For N=2, K=2: 1, 2, 2, 1. (1->2, 2->1, reversed: 1, 2, 2, 1).
    # For N=6, K=1: 3, 6, 5, 4, 2, 1. (Wait, Sample 3 says 3 6 5 4 2 1).
    # Let's check Sample 3: N=6, K=1. S = 6! = 720. Target = 360.
    # The 360th permutation of (1,2,3,4,5,6).
    # 1... (120), 2... (120), 3... (120). 
    # The 360th is the last one starting with 3: (3, 6, 5, 4, 2, 1).
    # This matches the sample!
    
    # So the rule is:
    # For the first position, we want the largest d such that 
    # sum_{i=1}^{d-1} Count(K, ..., K-1, ..., K) < (S+1)//2.
    # Since all counts are equal (K), the number of sequences starting with any d is S/N.
    # Target T = (S+1)//2.
    # First digit d: T <= d * (S/N) => d >= T / (S/N) = (S+1)/(2 * S/N) = N/2 + 1/(2S/N).
    # So d = ceil(N/2).
    # For the first digit d = (N+1)//2, the new target is T' = T - (d-1)*(S/N).
    # T' = (S+1)//2 - ((N+1)//2 - 1)*S/N.
    
    # This looks like we can determine the sequence by maintaining the target T.
    # But we can't use loops. We can use a recursive function with a cache 
    # or a clever list comprehension. However, the total length is 250,000.
    # We must use a mathematical pattern.
    
    # For K=1, the middle permutation of (1...N) is the last permutation starting with (N+1)//2.
    # That is: ((N+1)//2, N, N-1, ..., 1) excluding ((N+1)//2).
    # For N=6, K=1: (6+1)//2 = 3. Sequence: 3, 6, 5, 4, 2, 1.
    # For N=2, K=2: S=6, T=3. 
    # d=1: count=6*(2/4)=3. T=3 <= 3, so d=1.
    # New T = 3. Remaining: {1:1, 2:2}. Total=3.
    # d=1: count=3*(1/3)=1. T=3 > 1, so T=3-1=2.
    # d=2: count=3*(2/3)=2. T=2 <= 2, so d=2.
    # New T = 2. Remaining: {1:1, 2:1}. Total=2.
    # d=1: count=2*(1/2)=1. T=2 > 1, so T=2-1=1.
    # d=2: count=2*(1/2)=1. T=1 <= 1, so d=2.
    # New T = 1. Remaining: {1:1, 2:0}. Total=1.
    # d