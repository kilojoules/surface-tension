```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The constraint to avoid explicit loops forces a functional programming style.
    # While not necessarily "cleaner" for this specific logic, it encourages 
    # the use of high-order functions and recursion.
    
    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # Note: If S is the total number of permutations of a multiset, 
    # the "middle" sequence is the one that is the lexicographical 
    # complement of the first sequence if we consider the symmetry 
    # of reversing the alphabet (1 <-> N, 2 <-> N-1, etc.) 
    # and reversing the sequence itself.
    # However, the problem asks for the floor((S+1)/2)-th.
    # For a multiset, the lexicographical middle is exactly the sequence 
    # that is the "reverse-complement" of the first sequence.
    # The first sequence is: 1 (K times), 2 (K times), ..., N (K times).
    # The last sequence is: N (K times), N-1 (K times), ..., 1 (K times).
    # The sequence at index (S+1)//2 is the one that is "halfway".
    # Due to the symmetry of the multiset permutations, the sequence at 
    # index (S+1)//2 is the one where we replace each element x with (N + 1 - x)
    # and then reverse the entire sequence, but only if we were looking for the 
    # mirror image. 
    # Actually, the property is: the sequence at rank R and the sequence at 
    # rank (S - R + 1) are "complements" (each element x replaced by N+1-x).
    # For R = (S+1)//2, if S is even, R = S//2. The R-th and (S-R+1)-th 
    # are complements.
    # The simplest way to find the middle sequence of a multiset is to 
    # realize that the sequence is the "reverse" of the first sequence 
    # if we mirror the values.
    # Wait, the property is simpler: the sequence at rank (S+1)//2 is 
    # the one that is its own complement when reversed? No.
    # Let's use the property: the sequence at rank (S+1)//2 is the 
    # lexicographical "median". For multisets, the median sequence is 
    # constructed by placing the elements in a specific balanced way.
    # Actually, for any multiset, the sequence at rank (S+1)//2 is 
    # the one where we list the elements in non-decreasing order, 
    # but we "flip" the logic.
    # Let's re-evaluate: 
    # Sample 1: N=2, K=2. S=6. (S+1)//2 = 3. Sequence: (1, 2, 2, 1).
    # Sample 3: N=6, K=1. S=720. (S+1)//2 = 360. Sequence: (3, 6, 5, 4, 2, 1).
    # In Sample 3, the 360th permutation of (1,2,3,4,5,6) is the last one 
    # starting with 3.
    # The last sequence starting with 3 is (3, 6, 5, 4, 2, 1).
    # This confirms the pattern: the (S+1)//2-th sequence is the 
    # lexicographically largest sequence that starts with the 
    # "middle" element.
    # If N is even, the middle elements are N//2 and N//2 + 1.
    # The total number of sequences starting with 1, ..., N//2 is exactly S/2.
    # So the (S/2)-th sequence is the largest sequence starting with N//2.
    # If N is odd, the middle element is (N+1)//2.
    # The number of sequences starting with 1, ..., (N-1)//2 is < S/2.
    # The number of sequences starting with 1, ..., (N+1)//2 is > S/2.
    # So the (S+1)//2-th sequence is some sequence starting with (N+1)//2.
    
    # Correct Logic:
    # The total number of sequences is S = (N*K)! / (K!)^N.
    # We want the rank R = (S+1)//2.
    # We determine the first element x by finding the smallest x such that
    # sum_{i=1 to x} (count of sequences starting with i) >= R.
    # The number of sequences starting with i is (N*K - 1)! / ((K-1)! * (K!)^(N-1)).
    # This is S * (K / (N*K)) = S / N.
    # So each starting digit 1...N has exactly S/N sequences.
    # R = (S+1)//2. The starting digit x is ceil(R / (S/N)) = ceil((S+1)//2 / (S/N)).
    # x = ceil((S+1)*N / (2*S)) approx N/2.
    # Specifically, x = (N + 1) // 2.
    # Once the first digit x is fixed, we need the rank R' = R - (x-1)*(S/N).
    # If x < (N+1)//2, we are looking for a sequence in the range of x.
    # If x = (N+1)//2 and N is even, R = S/2, so we want the very last sequence 
    # starting with N/2.
    # If x = (N+1)//2 and N is odd, R = (S+1)//2, we are looking for the 
    # (S/2 - (N-1)/2 * S/N)-th sequence starting with (N+1)//2.
    # This is the (S/2 - S/2)-th? No.
    # Let's use the symmetry: the sequence at rank R is the "complement" 
    # of the sequence at rank S-R+1.
    # The complement of a sequence is replacing each x with N+1-x.
    # We want R = (S+1)//2.
    # If S is even, we want rank S/2. Its complement is rank S - S/2 + 1 = S/2 + 1.
    # If S is odd, we want rank (S+1)/2. Its complement is rank S - (S+1)/2 + 1 = (S+1)/2.
    # It is easier to find the rank R' = S - R + 1 and then complement it.
    # R' = S - (S+1)//2 + 1 = S // 2 + 1.
    # For N=2, K=2: S=6, R=3. R' = 6-3+1 = 4.
    # Rank 4 is (2, 1, 1, 2). Complement: (1, 2, 2, 1). Correct.
    # For N=6, K=1: S=720, R=360. R' = 720-360+1 = 361.
    # Rank 361 is the first sequence starting with 4: (4, 1, 2, 3, 5, 6).
    # Complement: (3, 6, 5, 4, 2, 1). Correct.
    # For N=3, K=3: S=1680/6=280? No, 9!/(3!^3) = 362880 / 216 = 1680.
    # R = 840. R' = 1680 - 840 + 1 = 841.
    # Rank 841 is the first sequence starting with 2: (2, 1, 1, 1, 3, 3, 3, 2, 2).
    # Wait, the sample says (2, 2, 2, 1, 3, 3, 3, 1, 1).
    # Let's re-calculate.
    # The first sequence starting with 2 is (2, 1, 1, 1, 2, 2, 3, 3, 3).
    # The complement of (2, 1, 1, 1, 2, 2, 3, 3, 3) is (2, 3, 3, 3, 2, 2, 1, 1, 1).
    # Sample 4 output: 2 2 2 1 3 3 3 1 1.
    # This is the complement of (2, 1, 1, 1, 3, 3, 3, 2, 2).
    # (2, 1, 1, 1, 3, 3, 3, 2, 2) is the last sequence starting with 2 that 
    # has 1s before 2s and 3s.
    # Actually, the simplest observation:
    # The sequence at rank (S+1)//2 is the "middle" one.
    # For any multiset, the sequence at rank (S+1)//2 is the one 
    # obtained by taking the first sequence (1...1, 2...2, ..., N...N),
    # and for the first half of the sequence, we use the "smallest" 
    # available numbers, and for the second half, we use the "largest".
    # No, that's not it.
    # Let's use the property: the sequence at rank (S+1)//2 is the 
    # lexicographical "median".
    # For a multiset, the median is the sequence where we 
    # arrange the elements such that we use the "middle" available 
    # element as much as possible.
    # Looking at the samples:
    # N=2, K=2: 1 2 2 1
    # N=6, K=1: 3 6 5 4 2 1
    # N=3, K=3: 2 2 2 1 3 3 3 1 1
    # Pattern:
    # In N=3, K=3: The middle element is 2. It appears K times first.
    # Then the remaining elements are 1 (K times) and 3 (K times).
    # They are arranged as 1 3 3 3 1 1? No.
    # Wait, the sample 4 output is 2 2 2 1 3 3 3 1 1.
    # This is: (Middle element K times), then (Smallest element 1 time), 
    # then (Largest element K times), then (Smallest element K-1 times).
    # Let's check N=2, K=2: Middle is 1 and 2. 
    # If we take 1, then 2, 2, 1. (Middle 1, then Largest 2, 2, then Smallest 1).
    # If N=6, K=1: Middle is 3. 
    # Sequence: 3, then (Largest 6, 5, 4, 2, 1).
    # The pattern is:
    # 1. Start with the element x = (N+1)//2.
    # 2. Use it K times.
    # 3. Then use all elements > x in descending order, each K times.
    # 4. Then use all elements < x in descending order, each K times.
    # Let's test:
    # N=2, K=2: x=1. Sequence: 1(2), 2(2), 0(2) -> 1 1 2 2. Wrong.
    # Let's try: x = (N+1)//2.
    # N=2, K=2: x=1. Sample output 1 2 2 1.
    # N=6, K=1: x=3. Sample output 3 6 5 4 2 1.
    # N=3, K=3: x=2. Sample output 2 2 2 1 3 3 3 1 1.
    # Wait, Sample 4: 2 2 2 1 3 3 3 1 1.
    # This is: 2(3), 1(1), 