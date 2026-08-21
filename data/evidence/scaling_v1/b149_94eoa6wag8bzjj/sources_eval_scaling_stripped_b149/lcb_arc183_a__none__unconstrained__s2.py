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

    # The total number of good sequences is S = (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence T is the i-th,
    # its "complement" (where each value x is replaced by N - x + 1) 
    # and then reversed is not necessarily the (S-i+1)-th.
    # However, the complement sequence T' (where T'_i = N - T_i + 1)
    # is the (S-i+1)-th sequence.
    # The floor((S+1)/2)-th sequence is the one that is "just before" 
    # or exactly the middle.
    
    # For a sequence T, let complement(T) be the sequence where each element x 
    # is replaced by (N + 1 - x).
    # If T is the i-th sequence, complement(T) is the (S - i + 1)-th sequence.
    # We want i = floor((S+1)/2).
    # If S is even, i = S/2. The (S/2)-th sequence is the complement of the (S/2 + 1)-th.
    # If S is odd, i = (S+1)/2. The (S+1)/2-th sequence is its own complement.
    
    # A key property of the "middle" of all permutations of a multiset is that
    # it is the sequence that is lexicographically as close to the center as possible.
    # For the multiset {1*K, 2*K, ..., N*K}, the middle sequence is 
    # the one that starts with the "middle" available digit.
    
    # Specifically, the floor((S+1)/2)-th sequence is the one where we 
    # greedily pick the smallest possible digit such that the number of 
    # sequences starting with that digit (and all smaller digits) 
    # is at least floor((S+1)/2).
    
    # However, calculating S is impossible with large N, K.
    # We use the property: the middle sequence is the one that is 
    # "lexicographically balanced".
    # For N=2, K=2: S=6, floor(7/2)=3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
    # 3rd is 1221.
    # For N=6, K=1: S=720, floor(721/2)=360. 
    # The 360th permutation of (1,2,3,4,5,6) is (3, 6, 5, 4, 2, 1).
    
    # The pattern for the floor((S+1)/2)-th sequence is:
    # It is the sequence T such that T is the largest sequence that is 
    # lexicographically smaller than or equal to its complement T'.
    # This is achieved by:
    # For each position i from 1 to NK:
    # Try digits d = 1, 2, ..., N.
    # If we pick d, we need to check if the number of sequences starting with 
    # (prefix + d) is enough to reach the target index.
    
    # Instead of calculating S, we can use the fact that we want the 
    # "largest" sequence that is "smaller" than its complement.
    # This is equivalent to:
    # For the first index i where T_i != T'_i, we must have T_i < T'_i.
    # To make T as large as possible, we want T_i to be as large as possible,
    # but still T_i < N + 1 - T_i, so 2*T_i < N + 1.
    # Thus T_i = floor(N/2) if N is even, or (N-1)/2 if N is odd.
    # Wait, the sample 3 (N=6, K=1) gives 3 6 5 4 2 1.
    # Here T_1 = 3. N+1-T_1 = 7-3 = 4. 3 < 4.
    # Then for the remaining digits {1,2,4,5,6}, we want the largest possible 
    # sequence to stay just under the middle.
    # That means for the remaining positions, we pick the largest available digits.
    # But we must ensure the total sequence T < complement(T).
    # Since T_1 < T'_1 (3 < 4), the condition T < T' is already satisfied.
    # To make T the largest such sequence, we fill the rest of the positions 
    # with the largest available digits in descending order.
    # Let's check: T = (3, 6, 5, 4, 2, 1). 
    # Complement T' = (4, 1, 2, 3, 5, 6).
    # T < T' is true. Any sequence starting with 4... would be > T'.
    # Any sequence starting with 3... and having a larger suffix than (6,5,4,2,1)
    # is impossible since (6,5,4,2,1) is the largest permutation of the remaining.
    
    # General Algorithm:
    # 1. The first digit T_1 is floor((N + 1) / 2) if we want the largest T < T'.
    #    Wait, if N=2, K=2, floor(3/2)=1. T_1=1. Remaining: {1,2,2}.
    #    Largest suffix: 2, 2, 1. Result: 1, 2, 2, 1. Matches Sample 1.
    #    If N=3, K=3, floor(4/2)=2. T_1=2. Remaining: {1,1,1,2,2,3,3,3}.
    #    Largest suffix: 3, 3, 3, 2, 2, 1, 1, 1. 
    #    Wait, Sample 4 says 2 2 2 1 3 3 3 1 1. Let's re-evaluate.
    #    Sample 4: N=3, K=3. S = 9!/(3!^3) = 1680. Target = 840.
    #    Sequences starting with 1: 8!/(2! 3! 3!) = 560.
    #    Target 840 is in the range starting with 2.
    #    Remaining index = 840 - 560 = 280.
    #    Sequences starting with 2: 8!/(3! 2! 3!) = 560.
    #    We need the 280th sequence starting with 2.
    #    The total sequences starting with 2 is 560. 
    #    The 280th is exactly the floor(560/2)-th.
    #    This is a recursive problem.
    
    # Let f(counts, target) be the function.
    # counts: list of remaining counts for each digit 1...N.
    # target: the index of the sequence we want.
    
    # To avoid loops, we use a helper that calculates the number of permutations.
    # But we can't use math.factorial for the target index logic in a loop.
    # Actually, we can use a recursive-like structure with a list of indices.
    
    # Since we need to avoid loops, we can pre-calculate the target index 
    # for each position.
    # But the target index depends on the digit chosen.
    # The only way to do this without loops is to use the symmetry property.
    # The floor((S+1)/2)-th sequence is the one that is "just smaller" than 
    # its complement.
    # This means at the first index i where T_i != T'_i, we have T_i < T'_i.
    # To maximize T, we want T_i to be as large as possible such that T_i < N + 1 - T_i.
    # This means T_i = (N - 1) // 2 + 1 if N is odd? No.
    # If N=3, T_i < 4 - T_i => 2*T_i < 4 => T_i < 2. So T_i = 1.
    # But Sample 4 says T_1 = 2. Let's re-read.
    # Sample 4: N=3, K=3. S=1680. Target=840.
    # Starts with 1: 560 sequences.
    # Starts with 2: 560 sequences.
    # 560 + 1 = 561. 840 is the 280th sequence starting with 2.
    # 280 / 560 = 1/2.
    # So for the first digit, we pick the digit d such that 
    # sum(count(1...d-1)) < target <= sum(count(1...d)).
    # And for the remaining, we seek the (target - sum(count(1...d-1)))-th sequence.
    
    # The symmetry is: the target index for the first digit is floor((S+1)/2).
    # If S_d is the number of sequences starting with digit d:
    # Target index for digit d is target - sum(S_1 ... S_{d-1}).
    # If we are looking for the middle of the whole set, and S_1 = S_N, S_2 = S_{N-1}...
    # The middle will fall into the digit d = (N+1)//2.
    # And the target index for that digit will be (sum(S_1...S_{d-1}) + 1)//2 if d is the middle.
    # Actually, the target index for the middle digit is simply (S_d + 1) // 2.
    
    # Let's trace N=3, K=3:
    # S_1 = 560, S_2 = 560, S_3 = 560. Total S = 1680.
    # Target = 840.
    # d=1: 560 < 840.
    # d=2: 560 + 560 >= 840. So T_1 = 2.
    # New target = 840 - 560 = 280.
    # New S_d for T_2 (remaining: 1*3, 2*2, 3*3):
    # S_1 = 7!/(2! 2! 3!) = 210
    # S_2 = 7!/(3! 1! 3!) = 140
    # S_3 = 7!/(3! 2! 2!) = 210
    # Target 280:
    # d=1: 210 < 280.
    # d=2: 210 + 140 >= 280. So T_2 = 2.
    # New target = 280 - 210 = 70.
    # New S_d for T_3 (remaining: 1*3, 2*1, 3*3):
    # S_1 = 6!/(2! 1! 3!) = 60
    # S_2 = 6!/(3! 0! 3!) = 20
    # S_3 = 6!/(3! 1! 2!) = 60
    # Target 70:
    # d=1: 60 < 70.
    # d=2: 60 + 20 >= 70. So T_3 = 2.
    # New target = 70 - 60 = 10.
    # This matches Sample 4: 2 2 2 ...
    
    # To implement this without loops, we can use a recursive-like structure 
    # by processing the sequence length NK.
    # Since we can't use loops, we use `reduce`.
    # The state will be (current_counts, current_target, result_sequence).
    
    from math import factorial
    
