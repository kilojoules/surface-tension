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
    # This is equivalent to finding the sequence at index (S-1)//2 (0-indexed).
    # Note: If S is odd, (S+1)//2 is the middle element. 
    # If S is even, (S+1)//2 is the S/2-th element.
    # In both cases, the index is (S-1)//2.
    
    # We use a helper to calculate the number of permutations of a multiset.
    # Instead of calculating S explicitly, we can determine the sequence 
    # greedily by checking how many sequences start with 1, 2, ..., N.
    
    # To avoid loops, we use reduce to build the sequence.
    # State: (current_counts, current_index)
    # current_counts: list of remaining counts for each number 1...N
    # current_index: the 0-indexed rank we are looking for.
    
    # We need a way to calculate the number of permutations of the remaining elements.
    # Total permutations = (sum(counts))! / product(counts[i]!)
    # However, we only need to know if the current_index is less than the number of 
    # sequences starting with a specific digit.
    
    # Let f(counts) be the total permutations of the multiset.
    # The number of sequences starting with digit 'd' is f(counts - {d}).
    # f(counts - {d}) = [ (sum(counts)-1)! / product(counts[i]!) ] * counts[d]
    #                = f(counts) * counts[d] / sum(counts)
    
    # Since we cannot use loops or recursion, we pre-calculate factorials 
    # if necessary, but we can't even use them in a loop.
    # Actually, we can use a mathematical property:
    # The "middle" sequence of a symmetric distribution of permutations 
    # is often related to the reverse of the first sequence.
    # But the rank is specific. 
    
    # Let's use the property that the total number of sequences S is 
    # symmetric. The sequence at rank (S-1)//2 is the one that 
    # "balances" the possibilities.
    
    # For a fixed set of counts, the number of sequences starting with 
    # digit d is: (Total_Remaining - 1)! / ( product(c_i!) ) * c_d
    # This is proportional to c_d.
    
    # The total number of sequences is S. We want the one at rank (S-1)//2.
    # In the first position, we check if (S-1)//2 < (S * c_1 / Total)
    # If yes, the first digit is 1.
    # If no, we subtract (S * c_1 / Total) from the rank and check digit 2.
    
    # Since we can't use loops, we use reduce to iterate through the 
    # NK positions. In each position, we use another reduce to find the 
    # correct digit by iterating 1...N.
    
    # To handle the large numbers, we use the fact that we only need 
    # to compare the rank with the number of permutations.
    # We can maintain the rank as a fraction or use a common denominator.
    
    # Let's use the property: the digit d is chosen if 
    # sum_{i=1 to d-1} (f(counts) * c_i / Total) <= rank < sum_{i=1 to d} (f(counts) * c_i / Total)
    # This is equivalent to:
    # sum_{i=1 to d-1} c_i <= (rank * Total) / f(counts) < sum_{i=1 to d} c_i
    
    # Let R = rank / f(counts). 
    # In the first step, R = ((S-1)//2) / S approx 1/2.
    # The digit d is the one where the cumulative sum of c_i first exceeds R * Total.
    
    # We can track the "relative rank" R in the range [0, 1).
    # For the first digit:
    # Total = N*K, c_i = K.
    # We want the d such that sum_{i=1}^{d-1} K <= R * (N*K) < sum_{i=1}^{d} K
    # (d-1)*K <= R * N * K < d*K  =>  (d-1)/N <= R < d/N
    # Since R = (S-1)//2 / S, for large S, R is approx 0.5.
    # So d is approx N/2.
    
    # Let's refine R. 
    # If we pick digit d, the new rank is:
    # rank_new = rank - sum_{i=1}^{d-1} (f(counts) * c_i / Total)
    # The new f(counts_new) = f(counts) * c_d / Total
    # R_new = rank_new / f(counts_new)
    # R_new = (rank - sum_{i=1}^{d-1} (f(counts) * c_i / Total)) / (f(counts) * c_d / Total)
    # R_new = (rank / f(counts) - (sum_{i=1}^{d-1} c_i / Total)) / (c_d / Total)
    # R_new = (R - (sum_{i=1}^{d-1} c_i / Total)) * (Total / c_d)
    
    # Initial R = ((S-1)//2) / S.
    # For large S, R is slightly less than 0.5.
    # If S is even, R = (S/2 - 1) / S = 1/2 - 1/S.
    # If S is odd, R = ((S-1)/2) / S = 1/2 - 1/(2S).
    
    # We can use a helper function to calculate S using a generator expression and math.prod.
    import math
    
    # Calculate S = (N*K)! / (K!)^N
    # Using math.comb is safer: S = comb(NK, K) * comb((N-1)K, K) * ... * comb(K, K)
    # But we can't use a loop. We can use a generator expression.
    # S = math.prod(math.comb(i * k, k) for i in range(1, n + 1))
    # Wait, the formula is: S = (NK)! / (K!)^N
    # S = math.comb(n*k, k) * math.comb((n-1)*k, k) * ...
    # This is equivalent to:
    # S = math.prod(math.comb(j * k, k) for j in range(1, n + 1))
    # Actually: S = math.comb(n*k, k) * math.comb((n-1)*k, k) ... is wrong.
    # Correct: S = math.comb(n*k, k) * math.comb((n-1)*k, k) is not it.
    # The number of ways to partition NK into N groups of K is (NK)! / (K!)^N / N!
    # But the groups (the digits) are distinct, so it's (NK)! / (K!)^N.
    
    # Let's use the property that we want the (S-1)//2-th sequence.
    # This is the "middle" sequence.
    # For any sequence, its "complement" (replacing digit d with N-d+1) is also a good sequence.
    # The lexicographical order is reversed.
    # The middle sequence is the one that is its own complement, if it exists.
    # If S is even, the (S/2)-th and (S/2 + 1)-th are complements.
    # We want the (S/2)-th.
    
    # The middle sequence is simply the sequence where we place the digits 
    # in a balanced way.
    # For N=2, K=2: S=6. (S+1)//2 = 3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
    # 3rd is 1221.
    # For N=3, K=3: S=1680. (S+1)//2 = 840.
    # The 840th sequence is the one just before the "mirror" point.
    
    # Observation: The middle sequence is the one that starts with the 
    # "middle" digit and distributes others around it.
    # Actually, there is a simpler pattern.
    # The sequence is: 
    # For i from 1 to NK:
    # The digit is determined by the remaining counts and the current rank.
    # Since we need the middle one, we can start with R = 0.5 (approximately).
    
    # Let's use the R_new formula with R = 0.5.
    # If S is very large, the difference between (S-1)//2 / S and 0.5 is negligible
    # for the first many digits.
    # However, we must be precise.
    
    # Let's use the property: the middle sequence is the one that 
    # reads the same forwards and backwards if we map digit d to N-d+1.
    # For N=2, K=2: 1 2 2 1. (1->2, 2->1). Correct.
    # For N=6, K=1: 3 6 5 4 2 1. (Wait, the sample says 3 6 5 4 2 1).
    # Let's check: N=6, K=1. S=6! = 720. (S+1)//2 = 360.
    # The 360th permutation of (1,2,3,4,5,6).
    # 1... starts at 1, 2... starts at 120, 3... starts at 240.
    # 360th is the last permutation starting with 3.
    # That is 3 6 5 4 2 1. Correct.
    
    # So the strategy is:
    # 1. Calculate S = math.factorial(n*k) // (math.factorial(k)**n)
    # 2. Target rank T = (S - 1) // 2
    # 3. Use reduce to find the sequence.
    
    s = math.factorial(n * k) // (math.factorial(k)**n)
    target = (s - 1) // 2
    
    # State: (current_counts, current_rank)
    # We use a list for counts because it's mutable, but we must treat it 
    # carefully in reduce. Actually, we can use a tuple.
    
    def get_digit(state, _):
        counts, rank = state
        total_rem = sum(counts)
        
        # We need to find the digit d such that:
        # sum_{i=1}^{d-1} (f(counts-i) ) <= rank < sum_{i=1}^{d} (f(counts-i))
        # f(counts-i) = (total_rem-1)! / product(c_j!) * c_i
        # Let base = (total_rem-1)! / product(c_j!)
        # rank = base * (sum_{i=1}^{d-1} c_i) + remainder
        # digit d is the one where sum_{i=1}^{d-1} c_i <= rank/base < sum_{i=1}^{d} c_i
        
        # To avoid loops, we use another reduce to find the digit.
        # We iterate through digits 1...N.
        
        # We need 'base' inside the reduce.
        # base = f(counts) / total_rem
        # But we can't calculate f(counts) every time.
        # Actually, we can! Python handles large integers.
        
        # Let's calculate the number of permutations for the current multiset.
        # current_s = math.factorial(total_rem) // (math.prod(map(math.factorial, counts)))
        # base = current_s // total_rem
        