```python
import sys
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. 
    # Due to the symmetry of lexicographical order, if we replace 
    # each element x with (N + 1 - x), the order is reversed.
    # The middle sequence is the one that is "closest" to its own 
    # complement in terms of lexicographical rank.
    
    # However, a simpler observation for the floor((S+1)/2)-th sequence:
    # It is the sequence that would appear exactly in the middle.
    # For N=2, K=2, S=6, we need the 3rd.
    # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # The 3rd is (1,2,2,1).
    
    # The general strategy to find the m-th lexicographical permutation 
    # of a multiset is to determine the first element by checking how many 
    # permutations start with 1, then 2, etc.
    
    # Total permutations S = (n*k)! / (k!)^n
    # We need m = (S + 1) // 2
    
    # Since we cannot compute S directly (too large), we use the property that
    # the middle sequence is the one where we try to place the smallest 
    # available digits as late as possible, but not so late that we 
    # pass the halfway point.
    
    # Actually, the floor((S+1)/2)-th sequence is the one that is 
    # lexicographically just before or equal to the "complement" 
    # sequence (where each x is replaced by N+1-x).
    
    # Let's use the digit-by-digit construction.
    # At each position, we try digits d = 1, 2, ..., N.
    # The number of sequences starting with the current prefix + d is:
    # Count = (RemainingTotal)! / Product(RemainingCounts!)
    
    # We need to find the smallest d such that the sum of counts for 
    # 1...d is >= (S+1)//2.
    
    # Since we can't compute S, we can maintain the target rank m.
    # But m is too large. Let's use the symmetry:
    # The middle sequence is the one that "balances" the counts.
    # It is known that for this specific problem, the result is the 
    # sequence that is lexicographically the largest among those 
    # that are smaller than or equal to their complement.
    
    # A more robust approach: the middle sequence is the one that 
    # starts with the smallest digit d such that the number of 
    # sequences starting with 1...d-1 is < (S+1)//2.
    
    # Let f(counts) be the number of permutations of the multiset.
    # We want the m-th.
    # Note: S is symmetric. The m-th sequence is the complement of the 
    # (S - m + 1)-th sequence.
    # For m = (S+1)//2, S-m+1 is either m or m+1.
    
    # The sequence we are looking for is the one that is "halfway".
    # This is the sequence that starts with the digit d such that
    # the number of sequences starting with 1...d-1 is < S/2
    # and the number of sequences starting with 1...d is >= S/2.
    
    # Let's use the property: the middle sequence is the one that 
    # is the "largest" sequence that is still "smaller" than its 
    # complement.
    
    # For N=2, K=2: S=6, m=3. 
    # d=1: count = 3!/2!1! = 3. 
    # Since 3 >= 3, the first digit is 1.
    # Remaining: {1:1, 2:2}. m=3.
    # d=1: count = 2!/2! = 1. (1 < 3)
    # d=2: count = 2!/1!1! = 2. (1+2 >= 3). First digit is 2.
    # Remaining: {1:1, 2:1}. m = 3 - 1 = 2.
    # d=1: count = 1!/1! = 1. (1 < 2)
    # d=2: count = 1!/1! = 1. (1+1 >= 2). First digit is 2.
    # Remaining: {1:1, 2:0}. m = 2 - 1 = 1.
    # d=1: count = 0!/0! = 1. (1 >= 1). First digit is 1.
    # Result: 1 2 2 1.
    
    # To implement this without huge integers, we can use the fact that
    # we only need to compare the current rank with S/2.
    # We can maintain the rank as a fraction or use the symmetry.
    # However, Python handles arbitrarily large integers, so we can 
    # just compute S and the counts.
    
    import math
    
    # Using a helper to compute the number of permutations of a multiset
    # total! / (k1! * k2! * ...)
    # Since we need to do this often, we can use a formula.
    
    # We can't use a loop to compute S if N*K is 250,000.
    # But we only need to compute the counts for the digits we actually 
    # place.
    # Wait, the total number of elements is N*K. We must determine 
    # the digit for each of the N*K positions.
    # That's a loop of N*K. Inside, we check N digits.
    # Total complexity O(N^2 * K). With N, K = 500, N^2*K = 125 million.
    # This might be too slow for Python. Let's optimize.
    
    # We can use the formula: Count(d) = (Total-1)! / (k1! ... (kd-1)! ... kN!)
    # Count(d) = [Total! / (k1! ... kN!)] * (kd / Total)
    # Let S_current be the total permutations of the remaining elements.
    # The number of permutations starting with digit d is S_current * (count[d] / Total).
    
    # We need the m-th sequence where m = (S_total + 1) // 2.
    # We can track the "relative" rank.
    # Let current_rank be the rank within the current set of permutations.
    # Initially current_rank = (S_total + 1) // 2.
    
    # To avoid loops, we can use a generator or map, but we need to 
    # update the state (counts and current_rank).
    # A reduce function is perfect for this.
    
    from functools import reduce
    
    # Precompute factorials for the initial S
    # But we can't use a loop to precompute. We can use math.factorial.
    
    # Initial state: (current_counts, current_rank, total_elements)
    # current_counts: list of counts for each digit 1...N
    # current_rank: the rank we are looking for
    # total_elements: N*K
    
    # In each step of reduce:
    # 1. Calculate S_current = total! / product(c!)
    # 2. Find d such that sum(S_current * c_i / total) for i < d < current_rank
    # 3. Update current_rank and current_counts
    
    # However, calculating S_current in every step is O(N) and 
    # we do it N*K times. Total O(N^2 * K).
    # We can optimize: S_next = S_current * (count[d] / total)
    # But we need to find d first.
    
    # Let's use the property that we want the middle sequence.
    # The middle sequence is the one that is "balanced".
    # For a given set of counts, the digit d that splits the 
    # permutations into two halves is the one where
    # sum(count[i] for i < d) < total / 2 <= sum(count[i] for i <= d)
    
    # Let's test this hypothesis:
    # N=2, K=2. Total=4. counts=[2, 2].
    # d=1: count[0]=2. 2 < 4/2 is false. 2 <= 4/2 is true.
    # So d=1.
    # Remaining: counts=[1, 2], total=3.
    # d=1: count[0]=1. 1 < 3/2 is true.
    # d=2: count[0]+count[1]=3. 3 >= 3/2 is true.
    # So d=2.
    # Remaining: counts=[1, 1], total=2.
    # d=1: count[0]=1. 1 < 2/2 is false. 1 <= 2/2 is true.
    # So d=1. Wait, the sample says 1 2 2 1.
    # My manual trace: 
    # Pos 1: total=4, counts=[2,2]. sum(i<1)=0, sum(i<=1)=2. 0 < 2 <= 2. d=1.
    # Pos 2: total=3, counts=[1,2]. sum(i<1)=0, sum(i<=1)=1. 1 < 1.5.
    #        sum(i<2)=1, sum(i<=2)=3. 1 < 1.5 <= 3. d=2.
    # Pos 3: total=2, counts=[1,1]. sum(i<1)=0, sum(i<=1)=1. 0 < 1 <= 1. d=1.
    # This gives 1 2 1 2. But the answer is 1 2 2 1.
    
    # The issue is the rank is (S+1)//2.
    # For N=2, K=2, S=6, m=3.
    # Pos 1: d=1, count=3. 3 >= 3. So d=1.
    # Pos 2: total=3, counts=[1,2], m=3.
    #       d=1, count=1. 1 < 3.
    #       d=2, count=2. 1+2 >= 3. So d=2.
    # Pos 3: total=2, counts=[1,1], m=3-1=2.
    #       d=1, count=1. 1 < 2.
    #       d=2, count=1. 1+1 >= 2. So d=2.
    # Pos 4: total=1, counts=[1,0], m=2-1=1.
    #       d=1, count=1. 1 >= 1. So d=1.
    # Result: 1 2 2 1. Correct.
    
    # To implement this without loops, we can use a recursive-like 
    # structure via reduce, but we need to handle the rank.
    # Since we can't use loops, we can't use a while loop to find d.
    # But we can use a list comprehension and `next()` to find d.
    
    # The total number of permutations S can be very large, but 
    # Python handles it. The only constraint is the time limit 
    # and the "no loop" rule.
    
    # We can use a helper function for the multiset permutation formula.
    # To avoid repeated factorial calls, we can't really, but 
    # we can use the property that we only need the ratio.
    
    # Let's refine the reduce state: (current_counts, current_rank, total_elements)
    # In each step:
    # 1. Find d using next(d for d in 1...N if sum(counts[i] for i < d) * S_curr / total >= current_rank)
    