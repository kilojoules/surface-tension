```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!^N).
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the "middle" sequence.
    # Due to the symmetry of the set of all good sequences, 
    # if a sequence A = (a_1, a_2, ..., a_{NK}) is the i-th sequence,
    # then the sequence A' = (N+1-a_1, N+1-a_2, ..., N+1-a_{NK}) 
    # is the (S - i + 1)-th sequence.
    # The middle sequence (floor((S+1)/2)) is the one that is its own 
    # "complement" in terms of lexicographical rank, or the one 
    # just before the complement of the first sequence.
    
    # More simply: the set of all good sequences is symmetric.
    # The sequence that is the "middle" one is the one that 
    # starts with the middle possible leading digit.
    # If we list all sequences, the first S/2 start with digits 
    # smaller than the "average" and the last S/2 start with larger.
    
    # For a fixed N and K, the sequence that is the floor((S+1)/2)-th
    # is the one where we try to place the digits in a way that 
    # balances the lexicographical weight.
    # The specific pattern for the floor((S+1)/2)-th sequence is:
    # For i from 1 to N:
    #   If i < (N+1)/2: the K copies of i are placed as late as possible.
    #   If i > (N+1)/2: the K copies of i are placed as early as possible.
    #   If i == (N+1)/2: the K copies of i are placed in the middle.
    
    # Actually, the symmetry property implies that the middle sequence
    # is the one that is "lexicographically central".
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # S=6, floor(7/2)=3. The 3rd is (1,2,2,1).
    # For N=6, K=1: S=720, floor(721/2)=360.
    # The 360th sequence of permutations of (1,2,3,4,5,6) is (3,6,5,4,2,1).
    
    # The general construction for the floor((S+1)/2)-th sequence:
    # We determine the digits one by one. For the first position, we want to find
    # the smallest digit d such that the number of sequences starting with 
    # digits < d is less than floor((S+1)/2), and the number of sequences 
    # starting with digits <= d is >= floor((S+1)/2).
    
    # However, calculating S is impossible for N=500. 
    # We use the property: the middle sequence is the one that is 
    # "complementary" to itself in the sorted list.
    # The sequence is: 
    # For each x from 1 to N:
    #   If x < (N+1)/2: place K copies of x at the end of the available slots.
    #   If x > (N+1)/2: place K copies of x at the beginning of the available slots.
    #   If x == (N+1)/2: place K copies of x in the remaining slots.
    
    # Let's refine:
    # The middle sequence is the one that starts with the digit ceil(N/2).
    # If N is even, say N=2, ceil(2/2)=1. It starts with 1.
    # If N=6, ceil(6/2)=3. It starts with 3.
    # The pattern is:
    # 1. Digits > (N+1)/2 are placed in increasing order of their values, 
    #    but each block of K is placed as early as possible.
    # 2. Digit == (N+1)/2 (if exists) is placed.
    # 3. Digits < (N+1)/2 are placed in decreasing order of their values,
    #    each block of K placed as late as possible.
    
    # Correct logic for the middle sequence:
    # The sequence is constructed by:
    # For v = 1 to N:
    #   If v < (N+1)/2: these K copies will be at the end, in decreasing order of v.
    #   If v > (N+1)/2: these K copies will be at the beginning, in increasing order of v.
    #   If v == (N+1)/2: these K copies will be in the middle.
    
    # Wait, the Sample 3 (N=6, K=1) output is 3 6 5 4 2 1.
    # N=6, (N+1)/2 = 3.5. 
    # v < 3.5: 1, 2, 3. v > 3.5: 4, 5, 6.
    # The output 3 6 5 4 2 1 suggests:
    # The digit ceil(N/2) comes first, then digits N down to ceil(N/2)+1, 
    # then digits ceil(N/2)-1 down to 1.
    # Let's check Sample 1: N=2, K=2. ceil(2/2)=1.
    # Sequence: 1 (K times), then 2 (K times), then 1 (remaining K times)? No.
    # Sample 1 output: 1 2 2 1.
    # This is: 1 (one time), 2 (K times), 1 (K-1 times).
    
    # Let's re-evaluate: the middle sequence is the one that is 
    # "just smaller" than its complement.
    # The complement of (a_1, ..., a_{NK}) is (N+1-a_1, ..., N+1-a_{NK}).
    # The middle sequence is the one where we take the lexicographically 
    # first sequence (1*K, 2*K, ..., N*K) and its complement (N*K, ..., 1*K)
    # and find the sequence exactly in the middle.
    
    # The actual pattern for the floor((S+1)/2)-th sequence is:
    # 1. The first digit is ceil(N/2).
    # 2. Then K-1 copies of ceil(N/2).
    # 3. Then digits from N down to ceil(N/2)+1, each K times.
    # 4. Then digits from ceil(N/2)-1 down to 1, each K times.
    # 5. Then the remaining copies of ceil(N/2)? No.
    
    # Let's use the property: the middle sequence is the one that 
    # starts with the "middle" digit, and then follows the "largest" 
    # possible arrangement for the remaining digits to stay just 
    # below the halfway point.
    # For N=6, K=1: Middle digit is 3. Remaining: {1,2,4,5,6}. 
    # Largest arrangement: 6, 5, 4, 2, 1. Result: 3 6 5 4 2 1.
    # For N=2, K=2: Middle digit is 1. Remaining: {1:1, 2:2}.
    # Largest arrangement: 2, 2, 1. Result: 1 2 2 1.
    # For N=3, K=3: Middle digit is 2. Remaining: {1:3, 2:2, 3:3}.
    # Largest arrangement: 2, 2, 3, 3, 3, 1, 1, 1? No, Sample 4 says 2 2 2 1 3 3 3 1 1.
    # Wait, Sample 4: 2 2 2 1 3 3 3 1 1. 
    # That is: 2(K times), 1(1 time), 3(K times), 1(K-1 times).
    # This looks like: 
    # Mid digit M = (N+1)//2.
    # Sequence: M (K times), then 1 (1 time), then N (K times), then N-1 (K times)... 
    # then M+1 (K times), then M-1 (K times)... then 1 (K-1 times).
    # No, that's not it.
    
    # Let's use the logic: the middle sequence is the one that 
    # starts with the smallest digit `d` such that the number of 
    # sequences starting with digits `< d` is `< S/2`.
    # The number of sequences starting with digit `d` is:
    # (NK-1)! / ((K-1)! * (K!^(N-1)))
    # This is S * (K / NK) = S / N.
    # So the first S/N sequences start with 1, the next S/N start with 2, etc.
    # We want the floor((S+1)/2)-th.
    # The digit `d` is ceil(N/2).
    # After placing one `d`, we want the floor((S+1)/2 - (d-1)*S/N)-th 
    # sequence of the remaining digits.
    # This is a recursive problem. The "middle" of the remaining 
    # is always the "largest" possible sequence if the current 
    # index is the middle digit, and "smallest" if we are above.
    
    # Correct Pattern:
    # The middle sequence is:
    # For i = 1 to NK:
    #   Find the smallest d in {1...N} (with count > 0) such that
    #   count_smaller(d) < target <= count_smaller(d) + count_current(d)
    #   Since we want the middle, and each digit is symmetric:
    #   The digit is always (N+1)//2 if available, or the closest available.
    #   Actually, the middle sequence is simply:
    #   (N+1)//2 repeated K times, then 
    #   the remaining digits arranged to be the "middle" of their own set.
    
    # Let's observe Sample 4: N=3, K=3. Mid=2.
    # Output: 2 2 2 1 3 3 3 1 1
    # This is: 2(K), 1(1), 3(K), 1(K-1).
    # Wait, the sample output 4 is "2 2 2 1 3 3 3 1 1".
    # Let's check the digits: 2 appears 3 times, 3 appears 3 times, 1 appears 3 times.
    # It is a good sequence.
    # The pattern is: 
    # 1. Digit M = (N+1)//2 repeated K times.
    # 2. Then the remaining digits (1...M-1 and M+1...N) 
    #    arranged such that we take the "largest" possible 
    #    arrangement of the "smaller" digits and the "smallest" 
    #    of the "larger" digits? No.
    
    # Let's use the property: the middle sequence is the one that 
    # is its own complement if S is odd, or the one just before 
    # the complement of the first if S is even.
    # The complement of (a_1, ..., a_{NK}) is (N+1-a_1, ..., N+1-a_{NK}).
    # The middle sequence A satisfies: A is the largest sequence 
    # such that A <= complement(A).
    # For A to be <= complement(A), the first index i where they differ 
    # must have a_i < N+1-a_i.
    # To make A as large as possible:
    # 1. We want a_i = N+1-a_i for as many i as possible.
    #    This is only possible if N+1 is even and a_i = (N+1)/2.
    # 2. At the first index i where we cannot have a_i