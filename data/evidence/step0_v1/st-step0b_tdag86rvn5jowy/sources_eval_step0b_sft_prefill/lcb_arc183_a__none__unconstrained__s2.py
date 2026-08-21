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
    # Note: S is symmetric. If a sequence is (s1, s2, ..., s_{NK}), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # The lexicographical middle of a symmetric set of permutations is 
    # effectively the sequence that is "halfway" through.
    # Specifically, for any sequence A, its complement A' is such that 
    # if A is the i-th sequence, A' is the (S - i + 1)-th sequence.
    # The floor((S+1)/2)-th sequence is the one immediately preceding the 
    # point of symmetry, or the middle one if S is odd.
    
    # A key property of lexicographical order for these multisets:
    # The "middle" sequence is the one that starts with the "middle" available number
    # and then follows the "smallest" possible pattern for the remainder, 
    # but since we want floor((S+1)/2), we can use the property that 
    # the sequence consists of K copies of 1, K copies of 2 ... K copies of N.
    # The sequence at index S/2 is the one that is lexicographically 
    # just smaller than its own complement.
    
    # To find the floor((S+1)/2)-th sequence:
    # We can determine the elements one by one.
    # For the first position, we try digits d = 1, 2, ..., N.
    # The number of sequences starting with d is (NK-1)! / (K!^{N-1} * (K-1)!).
    # We keep track of the cumulative count and the target index.
    
    # However, S can be massive (500*500)! / (500!^500), far exceeding 64-bit integers.
    # We need to use the symmetry property:
    # The floor((S+1)/2)-th sequence is the one where we effectively 
    # treat the "alphabet" as 1, 2, ..., N and we want the sequence 
    # that is just before the halfway mark.
    
    # Let's use the property: The sequence at rank floor((S+1)/2) 
    # is the one where we fill the positions from 1 to NK.
    # For each position, we check if the number of sequences starting with 
    # the current prefix and the next available digit 'd' is less than or equal to 
    # the remaining rank.
    
    # Given the constraints and the "middle" requirement, 
    # the floor((S+1)/2)-th sequence is the one where we 
    # essentially "balance" the digits.
    # For a multiset permutation, the sequence at rank floor(S/2) 
    # is the one that is lexicographically smaller than or equal to its 
    # complement (where complement of d is N+1-d).
    
    # The sequence that is exactly in the middle (or just before) 
    # is constructed by:
    # For each position i from 1 to NK:
    # We want to pick the smallest d such that the number of sequences 
    # starting with the current prefix is > S/2.
    # But we can't compute S.
    
    # Observation: The complement of a sequence (s1, ..., sm) is (N+1-s1, ..., N+1-sm).
    # If s1 < N+1-s1, then (s1, ...) < (N+1-s1, ...).
    # The first half of sequences start with d < (N+1)/2.
    # The second half start with d > (N+1)/2.
    # If N is even, the first S/2 sequences start with 1, ..., N/2.
    # If N is odd, the middle sequences start with (N+1)/2.
    
    # Correct logic for floor((S+1)/2)-th:
    # This is the sequence which, when compared to its complement, is the smaller one.
    # To construct this:
    # For each position, we want to pick a digit 'd'.
    # If we pick d < (N+1)/2, we are in the first half of the remaining possibilities.
    # If we pick d > (N+1)/2, we are in the second half.
    # If we pick d = (N+1)/2, we are in the middle.
    
    # Let's maintain the counts of each digit.
    # For each position, we try d = 1, 2, ..., N.
    # We need to know if the rank of the current prefix is <= S/2.
    # Instead of calculating S, we can track the "relative" rank.
    # Let the current state be (counts of digits, relative_rank).
    # relative_rank is a float between 0 and 1.
    # For the first digit, the proportions are K/NK, K/NK, ...
    # The first K/NK fraction of sequences start with 1, the next K/NK with 2, etc.
    # The target rank is 0.5.
    
    # We can use a list to track remaining counts and a float for the rank.
    counts = [K] * N
    rank = 0.5
    result = []
    
    # To avoid loops, we use a recursive-like approach via a list comprehension 
    # or a custom reduction, but since we need to update state, we'll use a loop.
    # Python's `for` loop is allowed.
    
    # We need to find d such that:
    # sum_{j < d} (count[j] / total) <= rank < sum_{j <= d} (count[j] / total)
    # Then the new rank is (rank - sum_{j < d} (count[j] / total)) / (count[d] / total)
    
    # To avoid floating point precision issues with 250,000 iterations, 
    # we can't use floats. But wait, the problem asks for floor((S+1)/2).
    # This is exactly the sequence that is "smaller than or equal to" its complement.
    # This means at the first index i where s_i != (N+1)/2, we must have s_i < (N+1)/2.
    # But we can't have s_i < (N+1)/2 for all i.
    
    # Actually, the sequence is:
    # For each position, if we have not yet deviated from the "middle",
    # we pick d = (N+1)/2 if N is odd.
    # If we must deviate, we want to stay in the lower half.
    # The symmetry is: the sequence S_idx is the complement of S_{S - idx + 1}.
    # The floor((S+1)/2)-th sequence is the one that is lexicographically 
    # smaller than or equal to its complement.
    # This means at the first index i where s_i != (N+1)/2, we must have s_i < (N+1)/2.
    # To make this the LAST such sequence (the floor of the middle),
    # we want to stay at (N+1)/2 as long as possible, and then 
    # when we must deviate, we pick the largest possible digit that is still < (N+1)/2?
    # No, that's not right.
    
    # Let's use the property:
    # The sequence is composed of K copies of each digit.
    # To be the floor((S+1)/2)-th, we should fill the sequence such that 
    # we are as "large" as possible while remaining <= the complement.
    # This means:
    # 1. Fill the middle digit (N+1)/2 as much as possible.
    # 2. For other digits, we want to balance them.
    # Actually, the simplest way:
    # For i = 1 to NK:
    #   Find the smallest d such that the number of sequences starting with 
    #   (prefix, d) is > remaining_rank.
    # Since we can't use floats/bigints, we use the symmetry:
    # The sequence is: 
    # For each pair of digits (d, N+1-d), we have K of each.
    # To be the middle sequence, we should arrange them as:
    # (N+1)/2 (if exists), then (N/2, N/2+1) repeated K times, 
    # but in an order that is just below the halfway point.
    # The middle sequence is:
    # K copies of (N+1)/2 (if N is odd), 
    # then K copies of (N/2, N/2+1, N/2, N/2+1, ...) 
    # No, that's not it.
    
    # Let's use the property: the sequence is the one where we 
    # greedily pick d = (N+1)//2 if possible.
    # If we have to pick between d and N+1-d, we pick d if we are in the first half.
    # The floor((S+1)/2)-th sequence is the one that is 
    # "largest" among those that are <= their complement.
    # A sequence is <= its complement if at the first index i where s_i != N+1-s_i, s_i < N+1-s_i.
    # To make this largest:
    # We want s_i = N+1-s_i for as many i as possible.
    # This means s_i = (N+1)/2.
    # Once we run out of (N+1)/2, we want the remaining to be as large as possible,
    # but the first difference must be s_i < N+1-s_i.
    # So we pick s_i = N/2 for the first "differing" slot, and then 
    # the rest of the sequence as large as possible.
    
    # Correct Construction:
    # 1. Let mid = (N + 1) / 2.
    # 2. If N is odd, we can have K copies of mid.
    # 3. We have K copies of each pair {d, N+1-d} for d = 1, ..., N//2.
    # 4. To be the largest sequence <= its complement:
    #    - We first use all K copies of mid (if N is odd).
    #    - Then we look at the pairs {d, N+1-d}.
    #    - We want to delay the s_i < N+1-s_i as much as possible.
    #    - But the first difference determines it.
    #    - To be the largest sequence that is <= its complement,
    #      the first difference must be s_i < N+1-s_i.
    #      To make the sequence largest, we want this i to be as large as possible.
    #      However, we can just pick s_i = N+1-s_i for all i until we can't.
    #      But we can't, because we have K of each.
    #      The sequence that is exactly the middle is:
    #      (N+1)/2 repeated K times, then (N/2, N/2+1) repeated K times, 
    #      then (N/2-1, N/2+2) repeated K times... 
    #      Wait, the lexicographical middle of all permutations of a multiset 
    #      is simply the sequence obtained by taking the sorted elements 
    #      and arranging them in a specific way.
    
    # Let's use the property: The complement of the i-th sequence is the (S-i+1)-th.
    # The floor((S+1)/2)-th sequence is the one that is <= its complement and is maximal.
    # This means:
    # For i = 1 to NK:
    #   We want to pick s_i such that the resulting sequence is <= its complement.
    #   If we pick s_i < (N+1)/2, then for all j > i, we can pick s_j as large as possible.
    #   If we pick s_i > (N+1)/2, then for all j < i, we must have had s_j = (N+1)/2.
    #   But if s_i > (N+1)/2, then s_i > N+1-s_i, so the sequence is > its complement.
    #   Therefore, we must have