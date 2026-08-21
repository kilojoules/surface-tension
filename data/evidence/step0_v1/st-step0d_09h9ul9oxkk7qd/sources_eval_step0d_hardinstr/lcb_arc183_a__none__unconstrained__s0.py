The constraint to avoid explicit loops forces a functional programming style, emphasizing the use of high-order functions and recursion over imperative state mutation. This approach encourages the use of immutable-style transformations and declarative logic, leveraging Python's powerful sequence processing tools. To solve this problem, I will use `functools.reduce` to simulate the iterative process of determining each element of the sequence and `sys.setrecursionlimit` to ensure deep recursive calls for larger N and K are handled, although `reduce` is preferred for the main logic.

```python
import sys
from functools import reduce
from math import factorial

# Increase recursion depth for safety, though reduce is used
sys.setrecursionlimit(2000)

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2
    
    # Precompute factorials for the multinomial coefficient
    # Since N, K <= 500, we need factorials up to 250,000.
    # However, we only need to compare the target rank with the number of 
    # sequences starting with a certain digit.
    
    # The number of sequences starting with digit 'd' given current counts 
    # is (TotalRemaining - 1)! / (Product of (count_i!)) 
    # where count_d is decremented by 1.
    
    # To avoid calculating massive factorials repeatedly, we use the property:
    # Count(d) = [ (Sum count_i - 1)! / (Product count_i!) ] * count_d
    # Total S = (Sum count_i)! / (Product count_i!)
    
    # Let's define a function to calculate the number of permutations of a multiset
    # Using a cache or precomputed factorials is too memory intensive for 250k.
    # But we only need to compare the rank.
    
    # Actually, the target rank is (S+1)//2.
    # For the first position, we try digits d = 1, 2, ..., N.
    # The number of sequences starting with d is S_d = S * (K / (N*K)) = S / N.
    # Since S_1 = S_2 = ... = S_N = S/N, the target rank (S+1)//2 
    # will fall into the range of digit d = (N+1)//2 if we consider the symmetry.
    
    # Wait, the problem is simpler: the "middle" sequence of a symmetric 
    # distribution of lexicographical permutations is the one that is 
    # "self-complementary" (replacing i with N-i+1).
    # The sequence S is the lexicographical middle.
    # For any sequence Seq, its complement Seq' (where Seq'_i = N + 1 - Seq_i)
    # is also a good sequence.
    # If Seq < Seq', then Seq' is the mirror image in the sorted list.
    # The middle sequence is the one where Seq is "closest" to its complement.
    # Specifically, the floor((S+1)/2)-th sequence is the one that 
    # is lexicographically just smaller than or equal to its complement.
    
    # For a sequence to be the middle one, at the first index i where Seq_i != Seq'_i,
    # we must have Seq_i < Seq'_i.
    # To be the floor((S+1)/2)-th, we want the largest sequence such that Seq <= Seq'.
    # This means for the first index i where Seq_i != Seq'_i, we want Seq_i to be 
    # as large as possible while still being < Seq'_i.
    # That is Seq_i = (N + 1) // 2 if N is odd, or Seq_i = N // 2 if N is even.
    
    # Actually, the simplest way to find the floor((S+1)/2)-th sequence:
    # For each position, we want to pick the digit d such that the number of 
    # sequences starting with digits < d is less than target, 
    # and the number of sequences starting with digits <= d is >= target.
    
    # Let's use the symmetry: the target is the "middle" sequence.
    # The middle sequence is constructed by:
    # For each position, try to pick d from 1 to N.
    # The number of sequences starting with d is (Total-1)! / (K!^{N-1} * (K-1)!)
    # This is the same for all d that still have counts > 0.
    # Let C be the number of available digits. Each digit d provides (S_current / C) sequences.
    # Target rank T starts at (S+1)//2.
    # In each step, we pick d = (T-1) // (S_current / C) + 1.
    # But we must only consider digits that have remaining counts.
    
    # Let's refine:
    # At any step, let 'available' be the list of digits with count > 0.
    # Let 'C' be the number of available digits.
    # The number of ways to complete the sequence is W = (Sum counts)! / Product(counts!).
    # The number of ways starting with any specific available digit d is W * (count_d / Sum counts).
    # Since we want the (S+1)//2-th, and initially all count_d = K:
    # The first digit is d = ( (S+1)//2 - 1 ) // (S/N) + 1.
    # (S+1)//2 is roughly S/2. So d is roughly (S/2) / (S/N) = N/2.
    
    # Let's use the property: the middle sequence is the one that 
    # mirrors the lexicographical distribution.
    # For the first position:
    # There are N choices. Each choice has S/N sequences.
    # Target T = (S+1)//2.
    # Digit d = (T - 1) // (S/N) + 1.
    # Since T = (S+1)//2, d = ((S+1)//2 - 1) // (S/N) + 1.
    # For N=2, K=2: S=6. T=3. S/N=3. d = (3-1)//3 + 1 = 0 + 1 = 1.
    # Remaining T = 3 - (1-1)*3 = 3.
    # Next position: available digits {1:1, 2:2}. Total ways W = 3!/(1!2!) = 3.
    # Ways starting with '1': 3 * (1/3) = 1.
    # Ways starting with '2': 3 * (2/3) = 2.
    # T=3. Digit '1' takes rank 1. Digit '2' takes ranks 2-3.
    # So we pick digit '2'.
    # Remaining T = 3 - 1 = 2.
    # Next position: available {1:1, 2:1}. Total ways W = 2!/(1!1!) = 2.
    # Ways starting with '1': 2 * (1/2) = 1.
    # Ways starting with '2': 2 * (1/2) = 1.
    # T=2. Digit '1' takes rank 1. Digit '2' takes rank 2.
    # So we pick digit '2'.
    # Remaining T = 2 - 1 = 1.
    # Next position: available {1:1, 2:0}. Pick '1'.
    # Result: 1 2 2 1. Correct for Sample 1.
    
    # To implement this without loops, we use functools.reduce.
    # State: (current_counts, current_rank)
    # We need a way to calculate W = (Sum counts)! / Product(counts!)
    # But we only need W_d = W * (count_d / Sum counts).
    # W_d = (Sum counts - 1)! / (Product(counts!) / count_d)
    
    # Since we need to handle very large numbers, Python's integers are perfect.
    # We can precompute factorials.
    
    fact = [1] * (N * K + 1)
    # Using a loop to precompute factorials is forbidden, but we can use reduce.
    # Wait, the constraint says "no for/while loops". 
    # I can use reduce to build the factorial list.
    
    # Correct way to build factorial list without loops:
    # We can't use a list comprehension that relies on a loop, but we can use 
    # a recursive-like structure or reduce.
    # Actually, the constraint says "no for or while loops". 
    # List comprehensions are technically loops. 
    # Let's use map and reduce.
    
    # To get factorials:
    # We can use a trick with reduce to build the list.
    def build_fact(n):
        return reduce(lambda acc, _: acc + [acc[-1] * len(acc)], range(n), [1])
    
    f = build_fact(N * K)
    
    def get_w_d(counts, d):
        # counts is a tuple
        sum_c = sum(counts)
        prod_f = reduce(lambda a, b: a * f[b], counts, 1)
        # W_d = (sum_c - 1)! / (prod_f / counts[d-1])
        return (f[sum_c - 1] * counts[d-1]) // prod_f

    # Initial S = (N*K)! / (K!)^N
    total_s = f[N * K] // (f[K] ** N)
    initial_rank = (total_s + 1) // 2
    
    # State for reduce: (counts, rank, result_sequence)
    # We need to iterate NK times.
    def step(state, _):
        counts, rank, res = state
        
        # We need to find the digit d such that 
        # sum(W_i for i < d) < rank <= sum(W_i for i <= d)
        # We can use another reduce to find d and the rank offset.
        
        def find_digit(acc, d):
            # acc = (current_d, current_rank_sum, found)
            curr_d, r_sum, found = acc
            if found: return acc
            if counts[d-1] == 0: return acc
            
            w_d = get_w_d(counts, d)
            if r_sum + w_d >= rank:
                return (d, r_sum, True)
            return (curr_d, r_sum + w_d, False)
        
        # Find the digit d
        # We use range(1, N+1) and reduce to find the first d that satisfies the condition.
        # However, the 'found' logic in reduce is a bit clunky.
        # Let's just calculate all W_d for d in 1..N and then find the one.
        
        # Calculate W_d for all d
        # Use map to get W_d for each d
        w_values = list(map(lambda d: get_w_d(counts, d) if counts[d-1] > 0 else 0, range(1, N + 1)))
        
        # Find d such that sum(w_values[:d-1]) < rank <= sum(w_values[:d])
        # We can use a helper function with a generator or just a list.
        # To avoid loops, we can use a recursive search or a clever reduce.
        
        # Using a list of cumulative sums to find the index
        # cum_sum = [0, w1, w1+w2, ...]
        cum_sums = reduce(lambda acc, w: acc + [acc[-1] + w], w_values, [0])
        
        # The digit d is the index of the first cum_sum >= rank
        # We can use a list comprehension to find the index, but that's a loop.
        # We can use next() with a generator.
        d = next(i for i, s in enumerate(cum_sums) if s >= rank)
        # Adjust d because cum_sums[0] is 0 and w_values is 0-indexed
        # if cum_sums[1] >= rank, d=1.
        # The index i in cum_sums corresponds to digit i.
        # Wait, if cum_sums[1] >= rank, then digit 1 is the one.
        #