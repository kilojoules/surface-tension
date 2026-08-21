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

    # The total number of good sequences S is (N*K)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle element (or the one just before the middle)
    # of the lexicographically sorted list of all permutations of the multiset.
    # A key property of lexicographical order for multisets is that the 
    # "middle" sequence is the one that is its own complement 
    # (where complement means replacing i with N - i + 1) 
    # if we consider the symmetry of the set.
    # Specifically, if a sequence A is the i-th, then the sequence A' 
    # (where A'_j = N + 1 - A_j) is the (S - i + 1)-th.
    # The sequence we are looking for is the one where A is "just smaller" 
    # than or equal to its complement A'.
    # This means we want the lexicographically smallest sequence A such that
    # A >= complement(A).
    # Actually, the simplest way to find the middle sequence is to 
    # construct it greedily: at each position, we try the smallest available 
    # number v, and check if the number of sequences starting with the 
    # current prefix is enough to reach the target index.
    # However, S can be massive, so we cannot compute it directly.
    # The symmetry property implies that the middle sequence is the one 
    # that is "balanced". 
    # For a sequence A, let f(A) be the sequence where each element x is 
    # replaced by (N + 1 - x). 
    # The sequence we want is the one such that it is the first sequence 
    # in lexicographical order that is >= its own complement.
    
    # To achieve A >= f(A), we want the first index i where A[i] != f(A)[i]
    # to have A[i] > f(A)[i].
    # If A[i] = f(A)[i] for all i, then A = f(A).
    # To find the smallest A such that A >= f(A):
    # We try to keep A[i] = f(A)[i] for as long as possible.
    # But A[i] = f(A)[i] is only possible if A[i] = (N+1)/2.
    # If N is even, A[i] can never equal f(A)[i].
    # If N is odd, A[i] can be (N+1)//2.
    
    # Correct logic for the middle sequence:
    # We want the sequence A such that it is the smallest sequence satisfying
    # A >= f(A).
    # This means at the first index i where A[i] != f(A)[i], we must have A[i] > f(A)[i].
    # To minimize A, we want this index i to be as late as possible.
    # For all j < i, we must have A[j] = f(A)[j].
    # This is only possible if A[j] = (N+1)/2.
    # But we only have K copies of (N+1)/2.
    # So for the first (N*K - K) positions, we cannot have A[j] = f(A)[j] 
    # unless N is odd and we use the middle element.
    
    # Let's refine: we want the smallest A such that A >= f(A).
    # This means we want A to start with the smallest possible values,
    # but we must ensure that at some point it "tips" to be larger than its complement.
    # The most "balanced" sequence is one where we place numbers in pairs (x, N+1-x).
    # To be the smallest A such that A >= f(A), we should use the smallest 
    # available numbers as much as possible, but we must ensure that 
    # the first time we deviate from the complement, we go "up".
    
    # The middle sequence is simply the one where we list the numbers 
    # 1, 2, ..., N, each K times, and then we find the middle permutation.
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # Middle (3rd) is (1,2,2,1).
    # Note: (1,2,2,1) is the complement of (2,1,1,2).
    # The rule is: the middle sequence is the one that is "lexicographically"
    # balanced. We can construct it by placing 1s and Ns, then 2s and N-1s, etc.
    # For the middle sequence, we want to use the smaller numbers as much as 
    # possible at the start, but we are constrained by the fact that 
    # the total sequence must be >= its complement.
    
    # The construction:
    # For i from 1 to N//2:
    #   Place (i) K times, then (N+1-i) K times? No.
    #   The middle sequence is:
    #   For i from 1 to N//2:
    #     K times the pair (i, N+1-i) is not quite right.
    #   Actually, the pattern is:
    #   For i from 1 to N//2:
    #     K times the value i, then K times the value N+1-i.
    #   Wait, Sample 1: N=2, K=2 -> 1 2 2 1.
    #   Here i=1, N+1-i=2. Sequence: 1, 2, 2, 1.
    #   Sample 4: N=3, K=3 -> 2 2 2 1 3 3 3 1 1.
    #   Here middle is 2. 2 2 2 (middle), then 1 3 3 3 1 1 (balanced).
    #   Wait, the sample 4 output is 2 2 2 1 3 3 3 1 1.
    #   Let's analyze: N=3, K=3. Middle element is 2.
    #   It starts with all K copies of the middle element (2),
    #   then it follows with the smallest available (1) and largest (3)
    #   in a way that the remaining part is the smallest sequence 
    #   that is >= its own complement.
    #   Actually, the pattern is:
    #   1. If N is odd, the middle element is M = (N+1)//2.
    #      The sequence starts with K copies of M.
    #      Then it continues with the remaining N-1 elements.
    #   2. For the remaining pairs (i, N+1-i) for i = 1 to N//2:
    #      We want the smallest sequence A such that A >= f(A).
    #      This is achieved by:
    #      For i = 1 to N//2:
    #        K times the value i, then K times the value N+1-i.
    #      Wait, that's for the very first sequence.
    #      For the middle one, we need to flip the balance.
    #      The correct pattern for the middle sequence is:
    #      For i = 1 to N//2:
    #        K times the value i, then K times the value N+1-i.
    #      But we do this for i = 1, 2, ..., N//2.
    #      Then we reverse the order of the pairs?
    #      Let's check Sample 1: N=2, K=2. i=1. 1, 2, 2, 1.
    #      Sample 4: N=3, K=3. M=2. 2,2,2, then i=1: 1, 3, 3, 3, 1, 1.
    #      Wait, the sample 4 output is 2 2 2 1 3 3 3 1 1.
    #      That is: K*M, then K*1, then K*3, then K*1... no.
    #      It's 2 2 2, then 1, then 3 3 3, then 1 1.
    #      That is: K*M, then 1, then K*3, then (K-1)*1.
    #      Let's re-evaluate: 1 3 3 3 1 1 is the middle of all permutations of {1,1,1,3,3,3}.
    #      The permutations of {1,1,1,3,3,3} are:
    #      111333, 113133, 113313, 113331, 131133, ...
    #      Total permutations = 6!/(3!3!) = 20.
    #      The 10th (floor(21/2)) is 113331.
    #      So for N=3, K=3, the answer is (2,2,2) + (1,1,3,3,3,1).
    #      Wait, the sample output says 2 2 2 1 3 3 3 1 1.
    #      That is 2,2,2, 1, 3,3,3, 1,1.
    #      Let's check the index: 1, 3, 3, 3, 1, 1 is the 11th permutation of {1,1,1,3,3,3}.
    #      111333 (1), 113133 (2), 113313 (3), 113331 (4), 131133 (5), 131313 (6), 131331 (7), 133113 (8), 133131 (9), 133311 (10).
    #      Wait, 133311 is the 10th.
    #      So the sequence is K*M, then the 10th permutation of the rest.
    #      The 10th permutation of {1,1,1,3,3,3} is 1 3 3 3 1 1.
    #      And 10 is floor((20+1)/2).
    
    # General Algorithm:
    # The middle sequence of a multiset is the one that is its own 
    # "lexicographical complement" (replacing smallest with largest, etc.)
    # if such a thing exists, or the one just before it.
    # For a multiset with pairs (i, N+1-i), the middle sequence is 
    # constructed by:
    # For i = 1 to N//2:
    #   K times i, then K times (N+1-i), but mirrored.
    #   Actually, the middle sequence of {1^K, 2^K} is 1 2 2 1 (for K=2).
    #   The middle sequence of {1^K, 3^K} is 1 3 3 3 1 1 (for K=3).
    #   The pattern is: 1, (K copies of N+1-i), (K-1 copies of i).
    #   Wait, let's check K=2: 1, 2, 2, 1. (1 copy of 1, 2 copies of 2, 1 copy of 1).
    #   For K=3: 1, 3, 3, 3, 1, 1. (1 copy of 1, 3 copies of 3, 2 copies of 1).
    #   This is: 1, then K copies of (N+1-i), then K-1 copies of i.
    #   But this is only for the pair (1, N). What about (2, N-1)?
    #   They must be nested.
    #   For N=4, K=2:
    #   Pairs are (1,4) and (2,3).
    #   The middle sequence would be: 1, 2, 3, 3, 2, 4, 4, 1.