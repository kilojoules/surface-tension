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
    # This is the "middle" sequence. 
    # Due to the symmetry of the set of good sequences, 
    # the middle sequence is the one that is "balanced".
    # Specifically, if we replace each element x with (N + 1 - x),
    # the lexicographical order is reversed.
    # The sequence we are looking for is the one that is 
    # lexicographically "central".
    
    # For a given N and K, the sequence that splits the 
    # total permutations into two equal halves is the one 
    # where we use the available numbers in a way that 
    # we stay as close to the "average" as possible.
    
    # However, a simpler observation:
    # The total number of sequences is S.
    # The sequence at index (S+1)//2 is the one where 
    # we greedily pick the smallest possible number that 
    # keeps the remaining permutations >= the remaining index.
    
    # Since we cannot compute S (it's too large), we use the 
    # property that the middle sequence is the one that 
    # mirrors the distribution.
    
    # For N=2, K=2: S=6. Index=3. 
    # 1. (1,1,2,2)
    # 2. (1,2,1,2)
    # 3. (1,2,2,1) <- Answer
    
    # The pattern for the middle sequence is:
    # We want to pick the smallest digit i such that the number of 
    # sequences starting with digits < i is less than (S+1)//2,
    # and the number of sequences starting with digits <= i is >= (S+1)//2.
    
    # Let f(n, k) be the total permutations.
    # The number of sequences starting with digit i is f(n-1, k) * comb(nk-1, k-1).
    # Actually, the number of sequences starting with digit i is 
    # (NK-1)! / ((K-1)! * (K!)^(N-1)).
    # This is the same for all i from 1 to N.
    # So there are N blocks of size S/N.
    # The index (S+1)//2 falls into block i = ceil((S+1)//2 / (S/N)).
    # i = ceil((N+1)//2) = (N+1)//2.
    
    # Once the first digit is fixed as i = (N+1)//2, 
    # we need to find the ((S+1)//2 - (i-1)*S/N)-th sequence 
    # of the remaining digits.
    
    # This is a recursive problem. The "middle" of the 
    # whole set is the sequence that starts with the 
    # middle digit, and then follows with the "middle" 
    # of the remaining permutations.
    
    # The middle digit of {1...N} is (N+1)//2.
    # After picking (N+1)//2, we have a set of digits where 
    # one digit has count K-1 and others have count K.
    # To keep the sequence "central", we should then pick 
    # the digits in an order that balances the remaining.
    
    # Observation from Sample 3 (N=6, K=1): 
    # S = 6! = 720. Index = 360.
    # The 360th permutation of (1,2,3,4,5,6) is (3, 6, 5, 4, 2, 1).
    # Note: 3 is the floor((6+1)/2). Then it sorts the 
    # remaining (1,2,4,5,6) in descending order? 
    # Let's check Sample 1 (N=2, K=2): Index 3.
    # Starts with 1. Remaining: {1, 2, 2}. 
    # Sorted: (1,2,2), (2,1,2), (2,2,1). 
    # The 2nd of these is (2,1,2). Wait, Sample 1 says (1,2,2,1).
    # Let's re-evaluate.
    
    # The total number of sequences is S.
    # The first digit is i if (i-1)*S/N < (S+1)//2 <= i*S/N.
    # i = ceil((S+1)//2 / (S/N)) = ceil((N+1)//2) = (N+1)//2.
    # The remaining index is (S+1)//2 - (i-1)*S/N.
    # This is roughly S/2 - (i-1)*S/N.
    # If i = (N+1)//2, the remaining index is roughly 
    # S/2 - ((N-1)//2)*S/N = S/2 - (1/2 - 1/2N)*S = S/2N.
    # This is the START of the remaining sequences.
    # So we pick the smallest available digits for the rest?
    # No, Sample 1: 1 2 2 1. First digit 1. Remaining {1, 2, 2}.
    # Index is 3 - 0 = 3. The 3rd permutation of {1, 2, 2} is (2, 2, 1).
    # So the result is 1 followed by 2, 2, 1.
    
    # Let's refine:
    # First digit: i = (N + 1) // 2
    # Remaining: K copies of all digits except i, and K-1 copies of i.
    # We need the (S+1)//2 - (i-1)*S/N -th permutation.
    # Since (i-1)*S/N is the number of sequences starting with 1...i-1,
    # and each digit starts S/N sequences, the index within the i-th block is:
    # target = (S+1)//2 - (i-1)*S/N.
    # Since S is huge, we can't use it. But we know:
    # target / (S/N) = ((S+1)//2 - (i-1)*S/N) / (S/N)
    # = (S/2) / (S/N) - (i-1) = N/2 - (i-1).
    # For N=2, i=1: target/(S/N) = 2/2 - 0 = 1. (The 1st of the block).
    # Wait, Sample 1: N=2, K=2. S=6. Index=3.
    # i=1: 0 < 3 <= 3. So first digit is 1.
    # Remaining index: 3 - 0 = 3.
    # Permutations of {1, 2, 2}: (1,2,2), (2,1,2), (2,2,1).
    # 3rd is (2,2,1). Result: 1 2 2 1. Correct.
    
    # For N=6, K=1: S=720. Index=360.
    # i=1: 0 < 360 <= 120 (False)
    # i=2: 120 < 360 <= 240 (False)
    # i=3: 240 < 360 <= 360 (True). First digit is 3.
    # Remaining index: 360 - 240 = 120.
    # Permutations of {1,2,4,5,6}: 5! = 120.
    # 120th is the last one: (6,5,4,2,1).
    # Result: 3 6 5 4 2 1. Correct.
    
    # General Rule:
    # 1. First digit i = (N + 1) // 2.
    # 2. The remaining index is the very last permutation of the remaining multiset.
    # 3. The last permutation is simply the remaining digits sorted descending.
    
    # Let's check Sample 4: N=3, K=3. S = 9! / (3!)^3 = 362880 / 216 = 1680.
    # Index = (1680 + 1) // 2 = 840.
    # Block size = 1680 / 3 = 560.
    # i=1: 0 < 840 <= 560 (False)
    # i=2: 560 < 840 <= 1120 (True). First digit is 2.
    # Remaining index: 840 - 560 = 280.
    # Total remaining permutations: 8! / (2! * 3! * 3!) = 40320 / (2 * 6 * 6) = 40320 / 72 = 560.
    # We need the 280th permutation of {1,1,1, 2,2, 3,3,3}.
    # This is exactly the middle (560 / 2 = 280).
    # This suggests a recursive structure.
    
    # Let's use the property: the middle sequence of a symmetric set 
    # is the one that is its own "complement" (mirror image) 
    # if we swap digits x and N+1-x and reverse the string.
    # But the simplest observation is:
    # The middle sequence is the one that starts with the middle digit,
    # and the remainder is the middle sequence of the remaining digits.
    
    # For N=3, K=3:
    # 1. Mid digit is 2.
    # 2. Remaining: {1:3, 2:2, 3:3}.
    # 3. Mid of these: starts with mid digit 2.
    # 4. Remaining: {1:3, 2:1, 3:3}.
    # 5. Mid of these: starts with mid digit 2.
    # 6. Remaining: {1:3, 3:3}.
    # 7. Mid of these: starts with mid digit 1 (since 1 is the smaller of the two).
    # Wait, the logic is: always pick the digit that keeps the 
    # remaining permutations balanced.
    
    # Correct logic for "middle" of a multiset:
    # While digits remain:
    #   Find the digit 'd' such that the number of permutations 
    #   starting with digits < d is < (Total+1)//2 
    #   and starting with digits <= d is >= (Total+1)//2.
    #   Append 'd', update Total and the multiset.
    
    # Since we can't compute Total, we use the fact that 
    # the "middle" sequence is the one that is 
    # lexicographically the "median".
    # For a multiset, the median sequence is the one where 
    # we always pick the digit that is the median of the 
    # current multiset.
    
    # Let's trace N=3, K=3 with "pick median digit":
    # Multiset: {1,1,1, 2,2,2, 3,3,3}. Median: 2.
    # Multiset: {1,1,1, 2,2, 3,3,3}. Median: 2.
    # Multiset: {1,1,1, 2, 3,3,3}. Median: 2.
    # Multiset: {1,1,1, 3,3,3}. Median: 1 (or 3, but we need the 
    # floor((S+1)/2), so we lean smaller).
    # Actually, for {1,1,1, 3,3,3}, the total permutations are 
    # 6!/(3!3!) = 20. The 10th is (1,3,1,3,1,3) ? No.
    # The 10th of {1,1,1, 3,3,3} is (1, 3, 3, 3, 1, 1).
    # Let's check:
    # 1