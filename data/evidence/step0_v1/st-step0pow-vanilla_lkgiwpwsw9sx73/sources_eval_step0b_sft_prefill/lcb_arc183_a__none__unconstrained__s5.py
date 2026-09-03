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

    # The total number of good sequences S is (NK)! / (K!^N).
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (a1, a2, ..., aM), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # The lexicographical middle of a symmetric set of permutations is 
    # effectively the sequence that is "halfway".
    # Specifically, for any sequence A, let A' be the sequence where A'_i = N + 1 - A_i.
    # A < A' if and only if at the first index i where they differ, A_i < N + 1 - A_i.
    # This means A_i < (N+1)/2.
    
    # To find the floor((S+1)/2)-th sequence:
    # We can construct the sequence greedily. For each position, we try digits d = 1, 2, ..., N.
    # We need to know how many sequences start with the current prefix.
    # However, S can be enormous, so we cannot compute it directly.
    # But we know that the "middle" sequence is the one that is lexicographically 
    # immediately smaller than or equal to its own complement.
    # The complement of (S1, ..., SM) is (N+1-S1, ..., N+1-SM).
    # The sequence that satisfies this is the one where we try to pick the "middle" 
    # element for the first position, and then balance the rest.
    
    # Actually, there is a simpler property:
    # The floor((S+1)/2)-th sequence is the one where we treat the alphabet 
    # as being symmetric. 
    # For the first position, if we pick d < (N+1)/2, we are in the first half of S.
    # If we pick d > (N+1)/2, we are in the second half.
    # If N is odd and we pick d = (N+1)/2, we are in the middle.
    
    # Let's use the property: the sequence we want is the one that is 
    # "lexicographically middle".
    # For each position i from 1 to NK:
    # We want to pick the smallest d such that the number of sequences 
    # starting with (prefix, d) is enough to reach the target index.
    # Since we can't compute S, we use the symmetry:
    # The target index is (S+1)//2.
    # For the first element d:
    # If d < (N+1)/2, all sequences starting with d are definitely in the first half.
    # If d > (N+1)/2, all sequences starting with d are definitely in the second half.
    # If N is odd and d = (N+1)/2, the sequences starting with d are split in half.
    
    # Let f(counts) be the number of ways to arrange the remaining elements.
    # f(counts) = (sum(counts))! / product(counts!)
    # We want to find the sequence A such that the number of sequences B < A is (S-1)//2.
    
    # Let's maintain the current counts of each number 1...N.
    # At each step, we iterate d from 1 to N.
    # We need to compare the number of sequences starting with d against the remaining rank.
    # To avoid huge numbers, we can use the fact that we only care about the rank relative to S.
    # Let R be the current rank we are looking for (initially (S+1)//2).
    # For d = 1 to N:
    #   count_d = f(counts - {d})
    #   if R <= count_d:
    #     A_i = d, break
    #   else:
    #     R -= count_d
    
    # To handle the huge numbers, we can use Python's arbitrary precision integers.
    # The constraints N, K <= 500 mean NK = 250,000. 
    # We cannot compute (NK)! directly.
    # But wait, we can observe the symmetry:
    # The middle sequence is simply the one where we pick d = (N+1)//2 as much as possible?
    # No. Let's use the property:
    # The floor((S+1)/2)-th sequence is the one where we effectively 
    # "mirror" the construction.
    # For each position, we want to pick d such that we don't exceed the halfway point.
    # If we pick d < (N+1)/2, we are consuming sequences from the bottom.
    # If we pick d > (N+1)/2, we are consuming sequences from the top.
    # The "middle" sequence is the one that is its own complement if S is odd, 
    # or the smaller of the two middle ones if S is even.
    # This means for each position, we should try to pick d = (N+1)/2.
    # If N is even, the two middle elements are N/2 and N/2 + 1.
    # The floor((S+1)/2)-th sequence will start with N/2 and then be the "largest" 
    # possible sequence starting with N/2.
    
    # Correct logic for floor((S+1)/2)-th:
    # It is the sequence A such that A_i is the smallest integer that makes 
    # the number of sequences lexicographically smaller than A at least (S+1)//2.
    # Due to symmetry, the sequence is:
    # For i = 1 to NK:
    #   Find smallest d such that:
    #   (number of sequences starting with prefix + d) > (number of sequences starting with prefix + (N+1-d))
    # This is getting complex. Let's use the property:
    # The middle sequence is the one where we fill the slots with 
    # (N+1)//2 as much as possible, but balanced.
    # Actually, the simplest way to describe the floor((S+1)/2)-th sequence:
    # For each position, we want to pick d such that we stay just below the halfway mark.
    # Because of the symmetry (d <-> N+1-d), the middle sequence is the one where 
    # we pick d = (N+1)//2 if N is odd.
    # If N is even, the first half ends with sequences starting with N/2.
    # The largest sequence starting with N/2 is (N/2, N/2, ..., N/2, N, N, ..., N, N-1, ..., 1).
    
    # Let's re-evaluate:
    # The set of all good sequences is X.
    # Define a map phi: X -> X where phi(S)_i = N + 1 - S_i.
    # phi is a lexicographical reversing permutation.
    # The floor((S+1)/2)-th element is the largest S such that S <= phi(S).
    # S <= phi(S) iff at the first index i where S_i != phi(S)_i, we have S_i < phi(S)_i.
    # S_i < N + 1 - S_i  <=>  2 * S_i < N + 1  <=>  S_i < (N+1)/2.
    # If S_i = (N+1)/2 for all i, then S = phi(S).
    
    # To find the largest S such that S <= phi(S):
    # For each position i = 1 to NK:
    # We want to pick the largest d such that there exists some sequence starting with 
    # (prefix, d) that is <= its complement.
    # If we pick d < (N+1)/2, then S < phi(S) is already guaranteed regardless of the rest.
    # To make S largest, we should pick the largest d < (N+1)/2, and then make the rest 
    # of the sequence as large as possible.
    # But we can also pick d = (N+1)/2 (if N is odd).
    # If we pick d = (N+1)/2, we then need to ensure the remaining suffix is <= its complement.
    
    # Strategy:
    # For each position i:
    # 1. Try d = N, N-1, ..., 1.
    # 2. If d > (N+1)/2:
    #    This d makes S > phi(S). This is only allowed if we already decided S < phi(S) 
    #    at some previous index j < i.
    # 3. If d < (N+1)/2:
    #    This d makes S < phi(S). This is always allowed. Once we do this, 
    #    all subsequent elements should be as large as possible (N, N, ..., 1).
    # 4. If d = (N+1)/2:
    #    This doesn't decide yet. We continue to the next index.
    
    # We need to track if we have already committed to S < phi(S).
    # Let `committed_smaller` be a boolean.
    
    # Since we can't use loops/recursion, we use a list comprehension or map.
    # We need to maintain the state (counts, committed_smaller).
    
    # We can use a mutable state in a reduce function.
    initial_state = ([K] * N, False, [])
    
    def process_step(state, _):
        counts, smaller, res = state
        # Find the largest d (1-indexed) that we can pick.
        # d is at index d-1 in counts.
        
        # We want the largest d such that:
        # If not smaller:
        #    if d < (N+1)/2: OK, smaller becomes True
        #    if d == (N+1)/2: OK, smaller stays False
        #    if d > (N+1)/2: NOT OK
        # If smaller:
        #    any d with counts[d-1] > 0 is OK
        
        # To find the largest d:
        # If smaller: largest d with counts[d-1] > 0.
        # If not smaller: 
        #    Check if there is any d == (N+1)/2 with counts[d-1] > 0.
        #    If yes, we can pick d = (N+1)/2 and stay not smaller.
        #    But we want the largest d. 
        #    Actually, if we pick d < (N+1)/2, we can then pick the largest possible for the rest.
        #    If we pick d = (N+1)/2, we are restricted again.
        #    So we should first check if we can pick d = (N+1)/2 and still find a valid sequence.
        #    Wait, the "largest S such that S <= phi(S)" means:
        #    At the first index i where S_i != (N+1)/2, we must have S_i < (N+1)/2.
        #    To make S largest, we want to keep S_i = (N+1)/2 for as long as possible.
        #    Once we can't, we pick the largest d < (N+1)/2, and then fill the rest with 
        #    the largest available numbers.
        
        # Let mid = (N+1)//2.
        # If not smaller:
        #    If counts[mid-1] > 0:
        #        d = mid, smaller = False
        #    Else:
        #        d = max(j for j in range(1, mid) if counts[j-1] > 0), smaller = True
        # If smaller:
        #    d = max(j for j in range(1, N+1) if counts[j-1] > 0), smaller = True
        
        mid = (N + 1) // 2
        if smaller:
            d = max([j for j in range(1, N + 1) if counts[j-1] > 0])
            new_smaller = True
        else:
            if counts[mid-1] > 0:
                d = mid
                new_smaller = False
            else:
                d = max([j for j in range(1, mid) if counts[j-1] > 0])
                new_smaller = True
        
        counts[d-1] -= 1
        return (counts, new_smaller, res + [d])

    # Use reduce to simulate the loop
    final_state = reduce(process_step, range(N * K), initial_state)
    print(*(