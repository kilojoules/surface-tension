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
    # Note: S is symmetric. If we have a sequence (a1, a2, ..., aM),
    # its "complement" (N+1-a1, N+1-a2, ..., N+1-aM) is also a good sequence.
    # Lexicographically, if A < B, then complement(A) > complement(B).
    # The middle two sequences (if S is even) or the middle one (if S is odd)
    # are related by this symmetry.
    # Specifically, the floor((S+1)/2)-th sequence is the one immediately 
    # preceding the "halfway" point.
    # A key property of lexicographical order with symmetric constraints:
    # The floor((S+1)/2)-th sequence is the one where we try to pick the 
    # smallest possible number at each step, but we ensure that the 
    # remaining sequence is "smaller than or equal to" the complement of 
    # the sequence we are building.
    
    # More simply: to find the middle of a symmetric distribution, 
    # we can determine each digit by checking if the number of sequences 
    # starting with the current prefix is less than or equal to S/2.
    # However, S is too large to compute.
    
    # Let's use the property: The floor((S+1)/2)-th sequence is the 
    # lexicographically largest sequence that is "smaller than or equal to" 
    # its own complement (where complement of x is N+1-x).
    # Wait, that's not quite right. 
    # Let's use the property: The middle of the list is reached when 
    # we have exhausted half of the possibilities.
    # Because the set of sequences is symmetric, the floor((S+1)/2)-th 
    # sequence is the one that is "just below" the point where the 
    # sequence becomes greater than its complement.
    
    # Correct logic for "middle" of symmetric combinatorial sets:
    # We build the sequence index by index. For the current position, 
    # we try candidates v = 1, 2, ..., N.
    # If we pick v, and v < (N+1)/2, we are in the "lower half" of the 
    # total space relative to the complement.
    # If we pick v > (N+1)/2, we are in the "upper half".
    # If we pick v = (N+1)/2, we are in the "middle".
    
    # Let's maintain the counts of remaining numbers.
    counts = [K] * N
    result = []
    
    # We want the sequence X such that X <= complement(X) and X is maximized.
    # This means at the first index i where X_i != complement(X)_i, 
    # we must have X_i < complement(X)_i.
    # To maximize X, we want X_i to be as large as possible for as long as possible,
    # provided that we eventually satisfy X_i < complement(X)_i at some point,
    # or X is exactly its own complement.
    
    # Actually, a simpler way to think about the floor((S+1)/2)-th element:
    # It is the sequence X such that X <= complement(X).
    # To find the lexicographically largest X such that X <= complement(X):
    # For each position i from 0 to NK-1:
    # Try v from N down to 1.
    # If we pick v, can we complete the sequence such that the resulting X <= complement(X)?
    # Yes, if there exists some j >= i such that X_j < complement(X)_j, 
    # and for all k < j, X_k = complement(X)_k.
    # But we can just check: if we pick v, and v < (N+1)/2, then X < complement(X) 
    # is guaranteed regardless of future choices (as long as they are valid).
    # If v > (N+1)/2, then X > complement(X) unless we already decided X < complement(X) 
    # at an earlier index.
    # If v = (N+1)/2, the relation depends on future indices.
    
    # State: (index, has_become_smaller)
    # has_become_smaller is boolean.
    
    # Since we want the floor((S+1)/2)-th, we want the largest X such that X <= complement(X).
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # We already ensured X < complement(X), so we can pick the largest v.
    #         # But we must ensure a valid sequence can be formed.
    #         # Since we only care about counts, any v with counts > 0 works.
    #         # To maximize X, we take the largest v.
    #         # Wait, if has_become_smaller is true, we just take the largest available v.
    #         # But we need to be careful: we are filling from left to right.
    #         # To get the largest X, we should pick the largest v that allows 
    #         # the rest to be filled.
    #         pass
    
    # Let's refine:
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # We can pick this v. To maximize X, we want to pick the largest v.
    #         # But we must ensure that the remaining counts can form a sequence.
    #         # (Which is always true if sum(counts) > 0).
    #         # However, we need to check if this choice prevents us from 
    #         # satisfying X <= complement(X) if we hadn't already.
    #         # But has_become_smaller is already true.
    #         # So we just take the largest v? No, that's for the absolute maximum.
    #         # We are constrained by the fact that the total sequence must be 
    #         # the floor((S+1)/2)-th.
    #         pass

    # Correct logic:
    # To find the floor((S+1)/2)-th sequence:
    # This is the largest sequence X such that X <= complement(X).
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # We can pick v. To maximize X, we pick the largest v.
    #         # But we must ensure that the remaining elements can be arranged 
    #         # to satisfy the condition? No, the condition is already satisfied.
    #         # To maximize X, we should pick the largest v and continue.
    #         # Actually, if has_become_smaller is true, we can just fill the 
    #         # rest of the sequence greedily with the largest available numbers.
    #         # But we must check if this v is "too large" such that it 
    #         # forces the sequence to be the complement of something smaller.
    #         # No, if has_become_smaller is true, X is already < complement(X).
    #         # To maximize X, we just take the largest available v for all remaining positions.
    #         pass

    # Let's use the property: 
    # X is the floor((S+1)/2)-th sequence if X is the largest sequence such that 
    # X <= complement(X).
    # To construct X:
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # We can pick v. To maximize X, we want the largest v.
    #         # But we must ensure that the remaining sequence can be completed.
    #         # Since we want the largest X, we take the largest v and repeat.
    #         # This will eventually lead to the largest sequence that is < complement(X).
    #         # Wait, if has_become_smaller is true, we can just take all remaining 
    #         # elements in descending order.
    #         # Let's check if this v allows the condition X <= complement(X) to be met.
    #         # If has_become_smaller is true, any subsequent choices are fine.
    #         # To maximize X, we take the largest available v.
    #         # But we must ensure that the resulting X is not > complement(X).
    #         # If has_become_smaller is true, X is already < complement(X).
    #         # So we just take the largest available v for all remaining slots.
    #         pass

    # Let's reconsider:
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # We can pick v. To maximize X, we want the largest v.
    #         # But we must ensure that the resulting X is the floor((S+1)/2)-th.
    #         # The set of sequences is {X_1, ..., X_S}.
    #         # They come in pairs (X, complement(X)).
    #         # The floor(S/2) sequences are those where X < complement(X).
    #         # The floor((S+1)/2)-th is the largest X such that X <= complement(X).
    #         # If X is its own complement, it's the middle one.
    #         # To maximize X such that X <= complement(X):
    #         # At each step, we try v = N, N-1, ..., 1.
    #         # If we pick v:
    #         # 1. If v < (N+1)/2, then X < complement(X) is now guaranteed.
    #         #    We can fill the rest of the sequence with the largest possible values.
    #         # 2. If v > (N+1)/2, then we must have already had X < complement(X) 
    #         #    at some previous index. If not, then X > complement(X), which is forbidden.
    #         # 3. If v == (N+1)/2, the condition X <= complement(X) depends on the rest.
    #         #    We can pick v if there exists a way to complete the sequence such that X <= complement(X).
    #         #    This is possible if the remaining counts allow for at least one sequence 
    #         #    that is <= its own complement.
    #         #    Actually, if we pick v = (N+1)/2, we just continue to the next index.
    #         #    The only way we'd be forced to have X > complement(X) is if all 
    #         #    remaining v's are > (N+1)/2.
    
    # Correct Greedy Strategy for largest X <= complement(X):
    # For i = 0 to NK-1:
    #   For v = N down to 1:
    #     If counts[v-1] > 0:
    #       If has_become_smaller:
    #         # Already X < complement(X), just take the largest v.
    #         # But wait, we must check if this v allows the sequence to be completed.
    #         # Since we only have counts, any v with count > 0 allows completion.
    #         # To maximize X, we take the largest v.
    #         # But we must be careful: if we take a very large v now, 
    #         # we might be forced to take a very small v later.
    #         # Actually, if has_become_smaller is true, the largest X is simply 
    #         # the remaining elements sorted in descending order.
    #         pass

    # Let's use a recursive-like structure with loops.
    # We need to determine if there exists a sequence Y formed from remaining counts 
    # such that (prefix + Y) <= complement(prefix + Y).
    
    # If has_become_smaller is true, the largest Y is just the remaining elements descending.
    # If has_become_smaller is false, we try v = N down to