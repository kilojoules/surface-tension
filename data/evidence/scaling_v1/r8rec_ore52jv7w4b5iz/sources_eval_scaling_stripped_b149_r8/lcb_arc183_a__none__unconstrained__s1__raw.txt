```python
import sys
from itertools import accumulate

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence A is the i-th,
    # its "complement" A' (where each element x is replaced by N - x + 1)
    # is the (S - i + 1)-th sequence.
    # The sequence at index floor((S+1)/2) is the one that is "just before" 
    # or exactly the middle of the set.
    
    # For a sequence to be the floor((S+1)/2)-th, we can determine each 
    # position greedily. However, calculating S is too slow.
    # Observation: The middle sequence is the one that is "lexicographically 
    # balanced". 
    # Specifically, for N=2, K=2, sequences are:
    # 1 1 2 2
    # 1 2 1 2
    # 1 2 2 1 <--- 3rd (floor(7/2))
    # 2 1 1 2
    # 2 1 2 1
    # 2 2 1 1
    
    # The pattern for the floor((S+1)/2)-th sequence is:
    # We want the sequence that is just smaller than its complement.
    # A sequence A is smaller than its complement A' if at the first index i 
    # where they differ, A[i] < A'[i].
    # The middle sequence is the largest sequence A such that A <= A'.
    # This happens when we try to keep the sequence as "small" as possible 
    # for the first half and "large" for the second, but the constraint 
    # is the global rank.
    
    # Actually, the property is simpler: 
    # The floor((S+1)/2)-th sequence is the one where we fill the sequence
    # by picking the smallest available number that allows the remaining 
    # suffixes to cover at least half of the remaining permutations.
    
    # But we can't compute permutations. Let's use the symmetry:
    # The sequence A is the floor((S+1)/2)-th if it is the largest sequence 
    # such that A <= complement(A).
    # To maximize A while A <= complement(A):
    # At each position i, we want to pick the largest possible value v 
    # such that the resulting sequence can still satisfy A <= complement(A).
    # This is achieved by:
    # For the first NK/2 positions, we try to stay "balanced".
    # Actually, the simplest construction for the middle sequence is:
    # Fill the sequence with numbers from 1 to N.
    # To be the floor((S+1)/2)-th, we want the sequence to be as large as 
    # possible while remaining in the first half.
    # This means we want the first index i where A[i] != complement(A)[i] 
    # to have A[i] < complement(A)[i], and for all j < i, A[j] = complement(A)[j].
    # But A[j] = complement(A)[j] is only possible if N is odd and A[j] = (N+1)/2.
    
    # Correct logic for floor((S+1)/2)-th:
    # It is the sequence that is "just below" the point of symmetry.
    # The symmetry is A <-> complement(A).
    # If N=1, the only sequence is (1, ..., 1).
    # If N > 1:
    # The first element of the middle sequence will be (N+1)//2.
    # If N is even, the first element will be N//2, and we want the 
    # lexicographically largest sequence starting with N//2.
    # Wait, if N=2, K=2, S=6, target=3rd. Sequences:
    # 1 1 2 2
    # 1 2 1 2
    # 1 2 2 1 <- Target
    # The first element is 1. Among sequences starting with 1, there are 
    # (2*2-1)! / (1! * 2!) = 3. 
    # Since we want the 3rd, and there are 3 starting with 1, 
    # it is the largest sequence starting with 1.
    
    # General Rule:
    # The total number of sequences starting with 1, 2, ..., (N-1)//2 
    # is definitely less than S/2.
    # The number of sequences starting with 1, ..., N//2 is >= S/2.
    # So the first element is N // 2 if N is even, or (N+1) // 2 if N is odd?
    # Let's check N=2, K=2: N//2 = 1. Largest sequence starting with 1 is 1 2 2 1. Correct.
    # Let's check N=3, K=3: S = 9!/(3!^3) = 1680. Target = 840.
    # Sequences starting with 1: 8!/(2! 3! 3!) = 560.
    # Sequences starting with 2: 8!/(3! 2! 3!) = 560.
    # Total starting with 1 or 2: 1120.
    # Since 560 < 840 <= 1120, the first element is 2.
    # We need the (840 - 560) = 280th sequence starting with 2.
    # Total sequences starting with 2 is 560. 280 is exactly 560/2.
    # So we need the floor((560+1)/2)-th sequence starting with 2.
    
    # This is a recursive problem. 
    # Let f(n, k) be the floor((S+1)/2)-th sequence.
    # If n=1, return [1]*k.
    # Let count(n, k) be the number of sequences.
    # The first element is the smallest 'v' such that 
    # sum_{i=1}^{v} (sequences starting with i) >= S/2.
    # Since all 'i' have the same number of sequences (S/n),
    # v * (S/n) >= S/2  => v >= n/2.
    # So v = ceil(n/2).
    # The rank we need among sequences starting with v is:
    # rank = floor((S+1)/2) - (v-1)*(S/n).
    
    # Let's trace N=3, K=3:
    # v = ceil(3/2) = 2.
    # rank = floor(1681/2) - (1)*(1680/3) = 840 - 560 = 280.
    # Now we need the 280th sequence of {1:3, 2:2, 3:3}.
    # This is no longer a symmetric "good sequence" problem because counts differ.
    # However, the target rank 280 is exactly half of the total 560.
    # When the target rank is exactly half of the total, and the 
    # distribution of remaining elements is symmetric (count of 1s == count of 3s),
    # the result is the largest sequence that is <= its complement.
    
    # For the "middle" sequence of a symmetric distribution:
    # While we have elements:
    # 1. If we can pick a middle element (N+1)//2 and keep the rest symmetric, we do.
    # 2. To stay in the first half, we want to pick the largest possible 
    #    element that doesn't push us into the second half.
    # 3. The symmetry is: if we have counts {c1, c2, ..., cn}, the complement 
    #    is {cn, cn-1, ..., c1}.
    # 4. If the counts are symmetric (c_i = c_{n-i+1}), the middle sequence 
    #    is the one that is lexicographically largest among those A <= complement(A).
    # 5. This is achieved by:
    #    - As long as we can, pick the "middle" value (N+1)//2.
    #    - If we must pick something else, to keep A <= complement(A), 
    #      once we pick a value v < (N+1)//2, we can then pick the largest 
    #      possible values for the rest.
    #    - But we want the largest A such that A <= complement(A).
    #    - This means we want to pick v = (N+1)//2 as long as possible.
    #    - If we pick v > (N+1)//2, we immediately become > complement(A).
    #    - So we pick v = (N+1)//2 until its count is 0.
    #    - Then we are left with pairs of (1, N), (2, N-1), etc.
    #    - To keep A <= complement(A) and maximize A, we should pick 
    #      the larger of the pair as late as possible.
    #    - Actually, the pattern is: 
    #      1. Use all of the middle element (N+1)//2 first.
    #      2. Then for the remaining pairs (i, N-i+1), 
    #         we want to pick the larger one as late as possible.
    #         Wait, Sample 4: N=3, K=3 -> 2 2 2 1 3 3 3 1 1.
    #         Middle element 2 is used 3 times. Then 1 3 3 3 1 1.
    #         This is: 2, 2, 2, then (1, 3) pairs.
    #         The sequence 1 3 3 3 1 1 is the largest sequence A 
    #         such that A <= complement(A) using three 1s and three 3s.
    #         The complement of (1, 3, 3, 3, 1, 1) is (3, 1, 1, 1, 3, 3).
    #         (1, 3, 3, 3, 1, 1) < (3, 1, 1, 1, 3, 3).
    #         Is it the largest? Let's try (3, ...). No, that's > complement.
    #         So we must start with 1. Then we want the largest possible 
    #         suffix: 3 3 3 1 1.
    #         So the pattern is: 
    #         - All of the middle element (N+1)//2.
    #         - Then, for each pair (i, N-i+1) from i = 1 to (N-1)//2:
    #           - We must pick the smaller one (i) first to ensure A < complement(A).
    #           - Then we can pick all of the larger ones (N-i+1).
    #           - Then the remaining smaller ones (i).
    #         - But we must do this for all pairs.
    #         - For N=3, K=3: Middle is 2. Pair is (1, 3).
    #           Sequence: 2 2 2, then 1, then 3 3 3, then 1 1.
    #           Result: 2 2 2 1 3 3 3 1 1. Matches Sample 4!
    #         - For N=6, K=1: Middle is 3 (since N is even, we use N//2).
    #           Wait, if N is even, the "middle" is between N//2 and N//2 + 1.
    #           The first element is N//2.
    #           Then we want the largest sequence starting with N//2.
    #           That means the remaining elements (1, 2, 4, 5, 6) should be in 
    #           descending order: 6 5 4 2 1.
    #           Result: 3 6 5 4 2 1. Matches Sample 3!
    
    # Refined Algorithm:
    # 1. If N == 1: print 1 (K times).
    # 2. If N is even:
