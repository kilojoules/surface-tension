```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the sequence exactly at the "middle" of the lexicographical range.
    # A key property of lexicographical order for permutations of a multiset
    # is that the sequence R, where R_i is the "complement" of sequence S_i,
    # is the mirror image in the sorted list.
    # The complement of value x in the range [1, N] is (N + 1 - x).
    # If a sequence is (S_1, S_2, ..., S_{NK}), its lexicographical "opposite"
    # is (N+1-S_1, N+1-S_2, ..., N+1-S_{NK}) REVERSED.
    # However, the problem asks for the middle sequence.
    # For a symmetric distribution of permutations, the middle sequence 
    # is the one that is "self-complementary" in a sense, or the one that 
    # balances the smaller and larger halves.
    
    # Specifically, for these constraints, the middle sequence is constructed by:
    # 1. Filling the first half of the slots with the smallest available numbers
    #    as much as possible, but since we need the MIDDLE, we actually 
    #    distribute the numbers such that the sequence is "balanced".
    # The pattern for the middle sequence of this multiset is:
    # For i from 1 to N:
    #   The number i appears K times.
    #   The middle sequence is formed by placing K copies of 1, then K copies of 2...
    #   but shifted. Actually, the symmetry implies that the middle sequence
    #   is the one where we list numbers from 1 to N, but the "middle" 
    #   of the total permutations occurs when the first element is (N+1)//2.
    
    # Let's refine: The total number of sequences is S.
    # The sequences starting with 1 are S/N.
    # The sequences starting with 2 are S/N.
    # The middle index (S+1)//2 falls into the range of sequences starting with
    # the ceiling(N/2)-th smallest available number.
    
    # For N=2, K=2: S=6. (6+1)//2 = 3.
    # Starts with 1: (1,1,2,2), (1,2,1,2), (1,2,2,1) -> 3rd is (1,2,2,1).
    # For N=6, K=1: S=720. (721)//2 = 360.
    # Starts with 1: 120, Starts with 2: 120, Starts with 3: 120.
    # 360th is the last sequence starting with 3: (3, 6, 5, 4, 2, 1).
    
    # General Rule:
    # To find the middle sequence, we can use the property that the 
    # "middle" sequence is the one that is the lexicographical 
    # reverse-complement of itself.
    # Or more simply: 
    # For the first position, the middle falls in the range of the 
    # (N+1)//2-th distinct element.
    # Once that element is placed, we repeat the process for the remaining.
    
    # Correct logic for the middle sequence of a multiset:
    # The sequence is: 
    # For i = 1 to N:
    #   If i < (N+1)//2: place K copies of i at the end.
    #   If i > (N+1)//2: place K copies of i at the beginning.
    #   If i == (N+1)//2: place K copies of i in the middle.
    # Then reverse the logic to fit the "middle" requirement.
    
    # Actually, the pattern is:
    # The middle sequence is:
    # (N // 2 + 1) repeated K times, 
    # then (N // 2) repeated K times, ..., 1 repeated K times,
    # then (N // 2 + 2) repeated K times, ..., N repeated K times.
    # Wait, let's check Sample 1: N=2, K=2. (2//2 + 1) = 2. 
    # Sequence: 2, 2, 1, 1? No, Sample 1 says 1, 2, 2, 1.
    
    # Let's re-evaluate:
    # The middle sequence is the one that is its own "complement-reverse".
    # Complement of x is (N+1-x).
    # Sequence A is the middle if A == reverse(complement(A)).
    # For N=2, K=2: (1, 2, 2, 1). Complement: (2, 1, 1, 2). Reverse: (2, 1, 1, 2). 
    # Wait, (1, 2, 2, 1) reversed is (1, 2, 2, 1), complement is (2, 1, 1, 2).
    # The property is: Sequence S is the middle if S is the "median" of the set.
    # The construction is:
    # For i from 1 to N:
    #   If i < (N+1)/2, it's "small". If i > (N+1)/2, it's "large".
    #   The middle sequence is:
    #   [ (N+1)//2 ] * K, then [N]*K, [N-1]*K ... [ (N+1)//2 + 1 ]*K,
    #   then [1]*K, [2]*K ... [ (N+1)//2 - 1 ]*K.
    # Let's check N=2, K=2: (N+1)//2 = 1.
    # [1]*2, then [2]*2. Result: 1 1 2 2. Still not 1 2 2 1.
    
    # Let's use the property: the middle sequence is the one that 
    # is the reverse of the sequence obtained by replacing each x with (N+1-x)
    # and then finding the "first" sequence.
    # Actually, the simplest construction for the middle sequence is:
    # For i = 1 to N:
    #   If i is odd, place K copies of i at the end of the current result.
    #   If i is even, place K copies of i at the beginning of the current result.
    # Let's check N=2, K=2:
    # i=1 (odd): res = [1, 1]
    # i=2 (even): res = [2, 2, 1, 1] -> Not 1 2 2 1.
    
    # Correct logic:
    # The middle sequence is:
    # For i from 1 to N:
    #   If i is odd, we put K copies of i at the "outer" edges.
    #   Specifically:
    #   The sequence is constructed as:
    #   For i = 1, 3, 5... (ascending): place K copies of i at the ends.
    #   For i = 2, 4, 6... (descending): place K copies of i at the ends.
    # This is getting complex. Let's use the property:
    # The middle sequence is:
    # (N+1)//2 repeated K times, 
    # then (N+1)//2 + 1 repeated K times, ..., N repeated K times,
    # then (N+1)//2 - 1 repeated K times, ..., 1 repeated K times.
    # NO.
    
    # Let's look at Sample 3: N=6, K=1. Output: 3 6 5 4 2 1
    # N=6, K=1. Middle is 360th.
    # Sequences starting with 1: 120, 2: 120, 3: 120.
    # The 360th is the very last sequence starting with 3.
    # The last sequence starting with 3 is (3, 6, 5, 4, 2, 1).
    # This means:
    # 1. The first element is (N+1)//2.
    # 2. The remaining elements are the REVERSE of the lexicographically 
    #    smallest sequence using the remaining elements.
    # Let's check N=2, K=2: (2+1)//2 = 1.
    # First element: 1. Remaining: {1:1, 2:2}.
    # Smallest remaining: (1, 2, 2). Reverse: (2, 2, 1).
    # Result: (1, 2, 2, 1). MATCHES Sample 1!
    # Let's check N=3, K=3: (3+1)//2 = 2.
    # First element: 2. Remaining: {1:3, 2:2, 3:3}.
    # Smallest remaining: (1, 1, 1, 2, 2, 3, 3, 3).
    # Reverse: (3, 3, 3, 2, 2, 1, 1, 1).
    # Result: (2, 3, 3, 3, 2, 2, 1, 1, 1).
    # Sample 4 says: 2 2 2 1 3 3 3 1 1. 
    # Wait, Sample 4 is 2 2 2 1 3 3 3 1 1.
    # My logic: First element is 2. But it says 2 2 2...
    # That means the first K elements are (N+1)//2.
    # Then the remaining are the reverse of the smallest sequence.
    # Let's check N=3, K=3 again:
    # First K elements: 2, 2, 2.
    # Remaining: {1:3, 3:3}.
    # Smallest remaining: (1, 1, 1, 3, 3, 3).
    # Reverse: (3, 3, 3, 1, 1, 1).
    # Result: (2, 2, 2, 3, 3, 3, 1, 1, 1).
    # Sample 4: 2 2 2 1 3 3 3 1 1. Still not matching.
    
    # Let's re-read Sample 4: 2 2 2 1 3 3 3 1 1.
    # This is: K copies of (N+1)//2, then 1 copy of (N+1)//2 - 1, 
    # then K copies of (N+1)//2 + 1, then remaining.
    # Actually, the pattern is:
    # The middle sequence is the one that is its own reverse-complement.
    # A sequence S is its own reverse-complement if S_i + S_{NK-i+1} = N + 1.
    # For N=3, K=3: S_i + S_{9-i+1} = 4.
    # S_1=2, S_9=2. S_2=2, S_8=2. S_3=2, S_7=2.
    # S_4=1, S_6=3. S_5=2.
    # Sequence: 2 2 2 1 2 3 2 2 2. 
    # But we only have K=3 copies of 2. This sequence has 6 copies of 2.
    # The constraint is: each number appears EXACTLY K times.
    # If S_i + S_{NK-i+1} = N + 1, then for any x, the number of times x appears
    # must be the same as the number of times (N+1-x) appears.
    # This is already true since both appear K times.
    # For the middle element (if NK is odd), it must be (N+1)/2.
    # For N=3, K=3, NK=9. S_5 must be 2.
    # For i=1, 2, 3, 4: we need to choose S_i such that the sequence is 
    # the "middle" one.
    # The middle sequence is simply: