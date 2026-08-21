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
    # This is essentially the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence 
    # (s1, s2, ..., snk) is the i-th, then the sequence 
    # (N+1-s1, N+1-s2, ..., N+1-snk) is the (S+1-i)-th.
    # The floor((S+1)/2)-th sequence is the one that is "just before" 
    # or exactly the middle.
    
    # For N=1, there is only one sequence.
    if N == 1:
        print(*( [1] * K ))
        return

    # To find the floor((S+1)/2)-th sequence, we can use the property that
    # the set of all sequences is symmetric. The middle sequence is 
    # closely related to the sequence that starts with the "middle" possible 
    # first digit.
    
    # However, a simpler observation for this specific problem:
    # The floor((S+1)/2)-th sequence is the one that, when mapped 
    # x -> N+1-x, becomes the ceil((S+1)/2)-th sequence.
    # For large N, K, we cannot compute S. But we can determine the 
    # sequence digit by digit.
    
    # The target index is idx = (S + 1) // 2.
    # We use a helper to check if the target index is within the range 
    # of sequences starting with digit 'd'.
    # The number of sequences starting with d is (NK-1)! / ( (K-1)! * (K!)^(N-1) ).
    # This is S * (K / NK) = S / N.
    
    # Since we need the floor((S+1)/2)-th, and the total is S:
    # If we are looking for the middle, the first digit will be (N+1)//2 
    # if we distribute S/N blocks.
    
    # More precisely, the sequences are divided into N blocks of size S/N.
    # The target index (S+1)//2 falls into block b = floor(((S+1)//2 - 1) / (S/N)) + 1.
    # b = floor((S+1 - 2) / (2S/N)) + 1 approx floor(N/2) + 1.
    
    # Let's use the symmetry: the sequence we want is the "complement" 
    # of the ceil((S+1)/2)-th sequence.
    # For N=2, K=2: S=6. floor(7/2)=3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
    # 3rd is 1221.
    
    # The logic for the middle sequence:
    # It is the sequence that is lexicographically largest among those 
    # that are <= the "mirror" of itself.
    # This is achieved by:
    # For the first digit, we want the largest d such that the number of 
    # sequences starting with 1...d-1 is < (S+1)//2.
    # The number of sequences starting with 1...d-1 is (d-1) * (S/N).
    # (d-1) * S/N < (S+1)/2  => d-1 < (N/2) + (N/2S) => d-1 <= floor(N/2).
    # So d = floor(N/2) + 1.
    
    # Once the first digit d is fixed, we need the ( (S+1)//2 - (d-1)*S/N )-th
    # sequence of the remaining.
    
    # Instead of complex math, we can use the property:
    # The floor((S+1)/2)-th sequence is the one that is "just smaller" 
    # than its complement.
    # This is the sequence that starts with (N+1)//2, and then follows 
    # the "largest" possible pattern for the remaining digits to stay 
    # under the midpoint, OR starts with N//2 and is the largest.
    
    # Correct logic for floor((S+1)/2)-th:
    # It is the sequence that is the mirror of the ceil((S+1)/2)-th.
    # For N=2, K=2, S=6, target=3. Mirror of 3rd is (6-3+1)=4th.
    # 4th is 2112. Mirror is 1221.
    
    # For N=3, K=3, S=1680. target=840.
    # The 840th sequence starts with digit d=2 because 1*S/3 = 560 and 2*S/3 = 1120.
    # It is the (840 - 560) = 280th sequence starting with 2.
    # Since 280 / (S/3) = 280 / 560 = 0.5, it's the middle of the '2' block.
    
    # The pattern for the floor((S+1)/2)-th sequence is:
    # 1. First digit is (N+1)//2.
    # 2. The remaining sequence is the "largest" possible sequence 
    #    that is still <= the mirror of the whole sequence.
    # This simplifies to:
    # The first digit is (N+1)//2.
    # The remaining digits are filled by:
    # - All remaining digits smaller than (N+1)//2 in descending order.
    # - All remaining digits larger than (N+1)//2 in ascending order.
    # Wait, that's not quite right. Let's use the property:
    # The middle sequence is the one that is "almost" its own complement.
    # The sequence is: 
    # Digit (N+1)//2, then all digits > (N+1)//2 in ascending order, 
    # then all digits < (N+1)//2 in descending order.
    # Let's check N=2, K=2: (2+1)//2 = 1. Digits > 1: {2,2}. Digits < 1: {}.
    # Result: 1, 2, 2, 1. (Correct)
    # N=6, K=1: (6+1)//2 = 3. Digits > 3: {4,5,6}. Digits < 3: {2,1}.
    # Result: 3, 4, 5, 6, 2, 1. (Wait, Sample 3 says 3 6 5 4 2 1)
    # Sample 3: 3 6 5 4 2 1. 
    # This is: First digit (N+1)//2, then digits > (N+1)//2 in DESCENDING order,
    # then digits < (N+1)//2 in DESCENDING order.
    # Let's check N=2, K=2: 1, then {2,2} desc, then {} desc => 1, 2, 2, 1. (Correct)
    # Let's check N=3, K=3: (3+1)//2 = 2. 
    # First digit 2, then {3,3,3} desc, then {1,1,1} desc.
    # But we have K=3, so the first digit 2 is used once. 
    # The remaining 2s must be placed.
    # Sample 4: 2 2 2 1 3 3 3 1 1. 
    # This is: all (N+1)//2, then all digits < (N+1)//2 descending, 
    # then all digits > (N+1)//2 ascending? No.
    # Let's re-examine Sample 4: 2 2 2 1 3 3 3 1 1.
    # Digits: 2(3 times), 1(1 time), 3(3 times), 1(2 times).
    # This looks like: 
    # All of digit (N+1)//2, 
    # then one of digit (N+1)//2 - 1, 
    # then all of digit N, then all of digit N-1... 
    # No, that's not it.
    
    # Let's use the property: the floor((S+1)/2)-th sequence is the 
    # lexicographical mirror of the ceil((S+1)/2)-th.
    # The ceil((S+1)/2)-th sequence is the smallest sequence that is 
    # >= its own mirror.
    # For N=3, K=3, the mirror of 2 2 2 1 3 3 3 1 1 is 2 2 2 3 1 1 1 3 3.
    # Actually, the simplest way to get the floor((S+1)/2)-th is:
    # It is the sequence that starts with (N+1)//2, 
    # then follows with the largest possible sequence using the remaining digits,
    # BUT we must stay below the mirror.
    
    # Correct observation:
    # The sequence is:
    # 1. All K copies of (N+1)//2.
    # 2. All K copies of (N+1)//2 - 1, (N+1)//2 - 2, ..., 1 (in descending order).
    # 3. All K copies of N, N-1, ..., (N+1)//2 + 1 (in descending order).
    # Wait, Sample 4: 2 2 2 1 3 3 3 1 1.
    # That is: 2,2,2 (all of mid), then 1 (one of mid-1), then 3,3,3 (all of max), then 1,1 (rest of mid-1).
    # This is getting confusing. Let's use the most reliable method:
    # The sequence is the mirror of the ceil((S+1)/2)-th.
    # The ceil((S+1)/2)-th sequence is:
    # First digit is (N // 2) + 1.
    # Then, the remaining digits are arranged to make the sequence as small as possible,
    # provided it's >= its mirror.
    # Actually, the simplest construction for floor((S+1)/2)-th is:
    # - All K copies of (N+1)//2.
    # - All K copies of (N+1)//2 - 1, ..., 1 (descending).
    # - All K copies of N, N-1, ..., (N+1)//2 + 1 (descending).
    # Let's check Sample 4 again: 2 2 2 1 3 3 3 1 1.
    # My rule: 2 2 2, 1 1 1, 3 3 3. Mirror: 2 2 2, 3 3 3, 1 1 1.
    # The Sample 4 output is 2 2 2 1 3 3 3 1 1.
    # This is: All of mid, then one of (mid-1), then all of (max...mid+1) desc, then rest of (mid-1...1) desc.
    
    # Final attempt at the logic:
    # The sequence is:
    # 1. K copies of (N+1)//2
    # 2. 1 copy of (N+1)//2 - 1 (if it exists)
    # 3. K copies of N, K copies of N-1, ..., K copies of (N+1)//2 + 1 (descending)
    # 4. K-1 copies of (N+1)//2 - 1, K copies of (N+1)//2 - 2, ..., K copies of 1 (descending)
    # Check Sample 4: N=3, K=3. Mid=2.
    # 1. 2 2 2
    # 2. 1
    # 3. 3 3 3
    # 4. 1 1
    # Result: 2 2 2 1 3 3 3 1 1. MATCHES!
    # Check Sample 3: N=6