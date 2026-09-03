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
    # Note: S is symmetric. If a sequence is (s1, s2, ..., s_m), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # The lexicographical middle of a symmetric set of permutations 
    # is reached by picking the "middle" element for the first position, 
    # and then filling the rest in descending order to reach the halfway point.
    
    # Specifically, for the first position:
    # There are N choices. The first floor(N/2) choices (1 to floor(N/2)) 
    # will cover the first half of the total sequences.
    # The (floor(S/2) + 1)-th sequence starts with the (floor(N/2) + 1)-th smallest 
    # available number if we treat the distribution symmetrically.
    
    # Let's refine:
    # The total number of sequences is S.
    # The number of sequences starting with '1' is S/N.
    # The number of sequences starting with '1' or '2' is 2S/N.
    # We want the index idx = (S + 1) // 2.
    # The first digit x is the smallest integer such that (x * S/N) >= idx.
    # x = ceil(idx / (S/N)) = ceil((S+1)//2 / (S/N)) = ceil((N(S+1))/(2S)).
    # For large S, (N(S+1))/(2S) is very close to N/2.
    # If N is even, x = N // 2. But we must be careful.
    # If N=2, K=2, S=6. idx=3. S/N = 3. x = ceil(3/3) = 1. 
    # Wait, Sample 1: N=2, K=2 -> (1, 2, 2, 1). First digit is 1.
    # Sample 3: N=6, K=1 -> (3, 6, 5, 4, 2, 1). First digit is 3.
    # Sample 4: N=3, K=3 -> (2, 2, 2, 1, 3, 3, 3, 1, 1). First digit is 2.
    
    # General Rule for the "middle" of all permutations of a multiset:
    # To find the sequence at rank (S+1)//2:
    # For the first position, we want the digit 'x' such that the number of sequences 
    # starting with 1...x-1 is < (S+1)//2 and starting with 1...x is >= (S+1)//2.
    # Due to symmetry, the middle sequence is the one that is its own "complement" 
    # (where complement of x is N+1-x) but shifted.
    # Actually, the simplest way to get the (S+1)//2-th sequence is:
    # For each position i from 1 to NK:
    # Try digits d = 1, 2, ..., N.
    # If we pick d, the number of ways to fill the rest is (RemainingTotal)! / Product(RemainingCounts!).
    # This is too slow.
    
    # Observation: The middle sequence is the one where we pick the "middle" 
    # available digit for the first position, and then for all subsequent 
    # positions, we pick the "largest" available digit to stay just below 
    # or at the halfway mark.
    
    # Let's test this hypothesis:
    # N=2, K=2: S=6, idx=3. 
    # Pos 1: d=1. Ways=3. 3 >= 3? Yes. First digit 1. Remaining idx = 3.
    # Pos 2: d=1. Ways=1. 1 < 3? Yes. idx = 3-1=2. d=2. Ways=2. 2 >= 2? Yes. Second digit 2.
    # Pos 3: d=1. Ways=1. 1 < 2? Yes. idx = 2-1=1. d=2. Ways=1. 1 >= 1? Yes. Third digit 2.
    # Pos 4: d=1. Fourth digit 1.
    # Result: 1 2 2 1. Correct.
    
    # N=6, K=1: S=720, idx=360.
    # Pos 1: d=1(120), 2(120), 3(120). 360 >= 360. First digit 3.
    # Remaining idx = 360 - 240 = 120.
    # For the rest, we want the 120th sequence of {1,2,4,5,6}.
    # Since 120 is the maximum possible (5!), the rest must be in descending order: 6 5 4 2 1.
    # Result: 3 6 5 4 2 1. Correct.
    
    # N=3, K=3: S=1680/6=280? No, 9!/(3!^3) = 362880 / 216 = 1680. idx=840.
    # Pos 1: d=1. Ways = 8!/(3!^2 * 2!) = 40320 / 72 = 560.
    # 560 < 840. idx = 840 - 560 = 280.
    # Pos 1: d=2. Ways = 8!/(3!^2 * 2!) = 560.
    # 560 >= 280. First digit 2.
    # Remaining idx = 280. Remaining multiset: {1:3, 2:2, 3:3}.
    # We want the 280th sequence of this multiset.
    # Total sequences for this multiset is 560. 280 is exactly half.
    # To get the middle of a set, we can use the property:
    # The (S+1)//2-th sequence is the one where we effectively 
    # "flip" the lexicographical order halfway.
    
    # Correct logic:
    # For each position, we find the smallest d such that 
    # count(sequences starting with prefix + d) >= current_rank.
    # If we find that current_rank is exactly (TotalWays + 1) // 2,
    # then for all subsequent positions, we just pick the largest available digit.
    
    # To avoid big integers and loops, we use the symmetry:
    # The middle sequence is:
    # 1. Find the first digit 'd' such that the number of sequences starting with 1...d-1 
    #    is < (S+1)//2 and starting with 1...d is >= (S+1)//2.
    # 2. Once we find this 'd', the remaining sequence is the 
    #    (current_rank - count(1...d-1))-th sequence of the remaining multiset.
    # 3. If this new rank is exactly (RemainingTotalWays + 1) // 2, 
    #    then the remaining sequence is the "middle" one.
    # 4. Crucial observation: The middle sequence of any multiset is 
    #    the one that is "self-complementary" in a sense.
    #    Actually, if we want the (S+1)//2-th sequence, we can just 
    #    determine the first digit, and then for the rest, 
    #    we want the (rank)-th sequence.
    #    If rank == (TotalWays + 1) // 2, then the sequence is 
    #    the one where we pick the middle element and then the rest in descending order?
    #    No. Let's use the property:
    #    The (S+1)//2-th sequence is the one where we pick the 
    #    "middle" digit for the first position, and then the 
    #    "largest" possible digits for the remaining positions 
    #    to fill the first half of the total space.
    
    # Let's use the property: 
    # The (S+1)//2-th sequence is the one where we 
    # 1. Find d such that sum_{i=1}^{d-1} count(i) < (S+1)//2 <= sum_{i=1}^{d} count(i).
    # 2. The remaining sequence is the ( (S+1)//2 - sum_{i=1}^{d-1} count(i) )-th sequence.
    # 3. If we ever hit the case where we need the (TotalWays + 1)//2 - th sequence 
    #    of a multiset, we can just fill the rest with the largest available digits 
    #    and then the smallest, but that's complex.
    
    # Let's use the property: The middle sequence is the one that is 
    # lexicographically the "median".
    # For a multiset, the median sequence is the one where we 
    # arrange the elements in non-decreasing order, but the 
    # "middle" element of the sorted unique elements is handled specially.
    
    # Actually, the simplest way to get the (S+1)//2-th sequence:
    # For each position:
    #   Find d such that rank <= count(d).
    #   If rank <= (count(d) + 1) // 2:
    #     The remaining sequence must be the (rank)-th sequence of the remaining multiset.
    #   Else:
    #     The remaining sequence must be the (count(d) - rank + 1)-th sequence 
    #     of the remaining multiset, but mirrored (complementary).
    
    # This is still recursive. Let's use the property:
    # The (S+1)//2-th sequence is:
    # For the first position, pick d = (N + 1) // 2.
    # Then for all subsequent positions, pick the largest available digit.
    # But we must ensure we have enough of digit d.
    
    # Let's trace Sample 4: N=3, K=3. S=1680. idx=840.
    # d=1: count=560. 840 > 560.
    # d=2: count=560. 840 <= 560 + 560. First digit is 2.
    # New rank = 840 - 560 = 280.
    # New total = 560. New rank 280 is exactly (560 // 2).
    # When rank == Total // 2, the remaining sequence is the largest possible 
    # sequence that is smaller than the middle.
    # The largest possible sequence is the descending one.
    # For N=3, K=3, first digit 2, remaining {1:3, 2:2, 3:3}.
    # Descending: 3 3 3 2 2 1 1 1.
    # Let's check Sample 4: 2 2 2 1 3 3 3 1 1.
    # Wait, the sample output is 2 2 2 1 3 3 3 1 1.
    # This is: first digit 2, then the remaining are 2, 2, 1, 3, 3, 3, 1, 1.
    # This is not descending.
    
    # Let's re-evaluate: (S+1)//2 is the middle.
    # The complement of a sequence (s1, ..., sm) is (N+1-s1, ..., N+1-sm).
    # Lexicographically, if A < B, then complement(A) > complement(B).
    # The middle sequence is the one where A = complement(A) if S is odd.
    # If S is even, (S/2)-th and (S/2 + 1)-th are complements.
    # (S+1)//2 is the S/2-th sequence.
    # The S/2-th sequence is the complement of the (S/2 + 1)-th sequence.
    # The (S/2 + 1)-th sequence is the