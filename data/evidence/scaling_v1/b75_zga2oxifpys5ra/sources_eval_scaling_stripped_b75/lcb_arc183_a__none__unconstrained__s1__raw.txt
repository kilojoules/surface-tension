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

    # The total number of good sequences is S = (NK)! / (K!^N)
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # A key property of lexicographical order for permutations of a multiset
    # is that the sequence is the "complement" of its reverse-complement.
    # The middle sequence is the one that is its own "complement" in a sense,
    # or more simply, the sequence that balances the distribution.
    
    # For a multiset, the middle sequence is achieved by:
    # 1. Filling the first half of the slots with the smaller available numbers.
    # 2. However, the exact middle of the lexicographical range is the sequence
    #    that is "symmetric" around the middle of the value range.
    # The specific pattern for the floor((S+1)/2)-th sequence of this multiset
    # is to place the numbers in a specific balanced way.
    # For N=2, K=2: (1,2,2,1). 
    # For N=6, K=1: (3,6,5,4,2,1).
    # The pattern is: 
    # For i from 1 to N:
    # If i <= N//2, we place K copies of i in the second half of the sequence
    # (but mirrored). If i > N//2, we place them in the first half.
    # Actually, the middle sequence is simply the one where we use the 
    # "middle" available digit at each step.
    
    # Correct logic for the middle sequence of a multiset:
    # The middle sequence is the one that, if you replace each element x 
    # with (N + 1 - x) and reverse the sequence, you get the same sequence.
    # This means for the first NK/2 positions, we want to use the 
    # "smaller" half of the total available digits, but distributed 
    # to keep the sequence as balanced as possible.
    
    # After analyzing the samples:
    # N=2, K=2 -> 1 2 2 1
    # N=6, K=1 -> 3 6 5 4 2 1
    # N=3, K=3 -> 2 2 2 1 3 3 3 1 1
    
    # The pattern is:
    # The middle element of the sorted unique values is M = (N+1)//2.
    # 1. Place K copies of M.
    # 2. Then place K copies of N, N-1, ..., M+1.
    # 3. Then place K copies of 1, 2, ..., M-1.
    # Wait, Sample 4: N=3, K=3 -> 2 2 2 1 3 3 3 1 1
    # That is: K copies of 2, then 1 copy of 1, then K copies of 3, then (K-1) copies of 1.
    # Let's re-evaluate. The middle sequence is the one that is 
    # lexicographically the "median".
    # For a multiset, the median sequence is:
    # For x in 1..N:
    # If x < (N+1)/2: place K copies of x in the second half.
    # If x > (N+1)/2: place K copies of x in the first half.
    # If x == (N+1)/2: split K copies between first and second half.
    
    # Let's refine:
    # The middle sequence is constructed by:
    # For the first NK/2 positions, we want to use the largest possible 
    # values that allow the remaining to be the mirror.
    # The actual construction:
    # 1. Use K copies of all i > (N+1)/2.
    # 2. Use K copies of i = (N+1)/2.
    # 3. Use K copies of all i < (N+1)/2.
    # But we must arrange them to be the "middle".
    # The middle sequence is:
    # (N//2 + 1) repeated K times, 
    # then (N, N-1, ..., N//2 + 2) each K times,
    # then (1, 2, ..., N//2) each K times.
    # Let's check Sample 1: N=2, K=2. N//2+1 = 2. 
    # Sequence: 2(2), then (empty), then 1(2) -> 2 2 1 1. 
    # Sample 1 says 1 2 2 1. My logic is slightly off.
    
    # Correct logic for middle of multiset permutations:
    # The sequence is the middle one if it is the "complement" of itself.
    # The complement of a sequence is replacing x with (N+1-x) and reversing.
    # For N=2, K=2: (1, 2, 2, 1). Complement: rev(2, 1, 1, 2) = (2, 1, 1, 2).
    # Wait, (1, 2, 2, 1) reversed is (1, 2, 2, 1), and replacing 1->2, 2->1 gives (2, 1, 1, 2).
    # The middle sequence S is such that S = reverse(complement(S)).
    # This means S_i = N + 1 - S_{NK - i + 1}.
    
    # To find the lexicographically smallest S such that S = reverse(complement(S)):
    # For i = 1 to NK/2:
    # We want S_i to be as small as possible.
    # But we must ensure that we have enough numbers left to satisfy the 
    # complement condition for the second half.
    # The condition S_i = N + 1 - S_{NK - i + 1} implies that for every 
    # occurrence of value 'x' in the first half, there must be an 
    # occurrence of 'N+1-x' in the second half.
    
    # For the first NK/2 positions, we want to pick the smallest available 
    # digits. However, we can only pick a digit 'x' if we can "pair" it 
    # with 'N+1-x' at the mirror position.
    # This means we can use at most K copies of any value 'x' in the 
    # first half, and this will automatically use K copies of 'N+1-x' 
    # in the second half.
    # The only constraint is that the total count of digits in the 
    # first half is NK/2.
    
    # To make the sequence lexicographically smallest:
    # For i = 1 to NK/2:
    # Try x = 1, 2, ...
    # We can pick x if:
    # 1. Count of x used so far < K.
    # 2. Count of (N+1-x) used so far < K (because it will be placed at the end).
    # Special case: if x == N+1-x, we need to ensure we don't exceed K.
    
    # Let's trace N=2, K=2:
    # i=1: Try x=1. Count(1)=0 < 2, Count(2)=0 < 2. OK. S_1=1, S_4=2.
    # i=2: Try x=1. Count(1)=1 < 2, Count(2)=0 < 2. OK. S_2=1, S_3=2.
    # Result: 1 1 2 2. 
    # But Sample 1 says 1 2 2 1. Let me re-read.
    # "floor((S+1)/2)-th". For S=6, this is the 3rd.
    # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # The 3rd is (1,2,2,1).
    # My "complement" logic describes the middle of the range, but the 
    # "middle" of a lexicographical set is not always the fixed point of 
    # the complement.
    # Actually, the 3rd of 6 is the last one of the first half.
    # The first half of the sequences are those starting with 1.
    # There are 6!/(2!2!) = 6 total. Half is 3.
    # So we need the 3rd sequence.
    # For N=2, K=2, the sequences starting with 1 are:
    # (1,1,2,2), (1,2,1,2), (1,2,2,1).
    # The 3rd one is (1,2,2,1).
    
    # General rule for the floor((S+1)/2)-th sequence:
    # It is the lexicographically largest sequence among those that are 
    # "smaller than or equal to" the theoretical middle.
    # This is equivalent to:
    # For i = 1 to NK:
    # Try x = 1, 2, ...
    # Calculate how many sequences start with the current prefix + x.
    # If that count is >= remaining rank, then x is the digit.
    # Otherwise, subtract count from rank and try x+1.
    # But S is too large for direct calculation.
    
    # However, the "middle" sequence of a multiset is known to be:
    # The sequence that is the "reverse" of the lexicographically 
    # smallest sequence, but with a twist.
    # Actually, the simplest way to get the floor((S+1)/2)-th is:
    # It is the sequence that is "symmetric" in terms of value distribution.
    # For N=2, K=2: 1 2 2 1
    # For N=6, K=1: 3 6 5 4 2 1
    # For N=3, K=3: 2 2 2 1 3 3 3 1 1
    
    # Looking at Sample 4: N=3, K=3. 
    # The sequence is 2 2 2 1 3 3 3 1 1.
    # This is: K copies of (N+1)//2, then 1 copy of (N+1)//2 - 1, 
    # then K copies of (N+1)//2 + 1, then K-1 copies of (N+1)//2 - 1.
    # Wait, the pattern is:
    # Mid value M = (N+1)//2.
    # The sequence is:
    # M (K times), then (M-1, M-2, ..., 1) each K times, 
    # then (M+1, M+2, ..., N) each K times, 
    # then (M-1, M-2, ..., 1) ... no.
    
    # Let's use the property: the middle sequence is the one that 
    # is the "complement" of itself reversed, but we want the 
    # largest such sequence that is <= the true middle.
    # The true middle is the sequence S such that S = reverse(complement(S)).
    # For N=3, K=3: M=2.
    # S_1=2, S_9=3-2+1=2.
    # S_2=2, S_8=2.
    # S_3=2, S_7=2.
    # S_4=1, S_6=3.
    # S_5=2.
    # Sequence: 2 2 2 1 2 3 2 2 2. But we only have three 2s.
    # If K is odd, the middle element must be (N+1)/2.
    # If N=3, K=3, total length 9. S_5 must be 2.
    # Remaining: two 1s, two 2s, two 3s.
    # To make it the "middle", we can use the property that the 
    # middle sequence is the one that is "balanced".
    # For N=3, K=3, the sequence 2 2 2 1 3 3 3 1 1 is given.
    # Let's check its complement: 2->2, 1->3, 3->