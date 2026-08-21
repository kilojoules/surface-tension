```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. 
    # A key property of lexicographical order for permutations of a multiset 
    # is that the sequence S_i is the complement of sequence S_{S-i+1} 
    # if we replace each element x with (N + 1 - x).
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, for the floor((S+1)/2)-th sequence, we want the sequence 
    # that is lexicographically just smaller than or equal to its complement.
    
    # The complement of a sequence (a_1, a_2, ..., a_{NK}) is (N+1-a_1, ..., N+1-a_{NK}).
    # If a sequence is the "middle" one, it should be as balanced as possible.
    # The sequence that is exactly in the middle (or the one just before the midpoint)
    # is constructed by placing the numbers in a specific balanced way.
    
    # For N=1, the only sequence is (1, ..., 1).
    # For N > 1, the middle sequence starts with the middle value of the available digits.
    # If N is even, the first digit of the floor((S+1)/2)-th sequence is N//2.
    # If N is odd, the first digit is (N+1)//2.
    # However, a simpler pattern emerges:
    # The sequence is constructed by taking the digits in the order:
    # (N//2) repeated K times, then (N//2 - 1) repeated K times... down to 1,
    # then (N//2 + 1) repeated K times... up to N.
    # Wait, that's not quite right. Let's re-evaluate.
    
    # The symmetry is: Sequence A is the i-th, Sequence B is the (S-i+1)-th.
    # B is A with each element x replaced by (N+1-x).
    # We want the sequence where A <= B lexicographically and i is maximized.
    # This happens when the first index j where A_j != B_j has A_j < B_j,
    # and for all m < j, A_m = B_m.
    # A_m = B_m implies A_m = N+1 - A_m, so 2*A_m = N+1.
    # This can only happen if N is odd and A_m = (N+1)//2.
    
    # If N is even, A_1 cannot be B_1. For A < B, we need A_1 < B_1.
    # To maximize i, we want A_1 to be as large as possible such that A_1 < N+1 - A_1.
    # So A_1 = N // 2.
    # Once A_1 is fixed at N // 2, to make the sequence as large as possible 
    # (to reach the middle), we want the remaining sequence to be as large as possible.
    # The largest possible remaining sequence is the reverse of the smallest.
    
    # Correct logic for the middle sequence:
    # 1. If N is odd, the first (N+1)//2 * K elements are not necessarily (N+1)//2.
    # Actually, the middle sequence is:
    # All (N+1)//2 are placed first? No.
    # Let's use the property: the middle sequence is the one that, when 
    # mirrored (x -> N+1-x), is its own lexicographical opposite.
    # The sequence is: 
    # For i from 1 to N:
    # If i < (N+1)/2, it appears K times at the end (in increasing order of i).
    # If i > (N+1)/2, it appears K times at the beginning (in decreasing order of i).
    # If i == (N+1)/2, it appears K times in the middle.
    
    # Let's refine:
    # The sequence is: 
    # (N, N, ..., N) [K times], (N-1, ..., N-1) [K times], ..., 
    # ((N+1)//2 + 1, ..., (N+1)//2 + 1) [K times],
    # ((N+1)//2, ..., (N+1)//2) [K times],
    # (1, ..., 1) [K times], (2, ..., 2) [K times], ..., 
    # (N//2, ..., N//2) [K times].
    # Wait, the sample 3: N=6, K=1 -> 3 6 5 4 2 1.
    # N=6, (N+1)//2 = 3. 
    # Sequence: 3, then 6, 5, 4, then 2, 1.
    # This matches! 
    # Sample 1: N=2, K=2 -> 1 2 2 1.
    # N=2, (N+1)//2 = 1.
    # Sequence: 1 (K times), then 2 (K times), then ... 
    # No, Sample 1 is 1 2 2 1. 
    # Let's look at Sample 1 again: 1 2 2 1.
    # N=2, K=2. Middle is 3rd of 6.
    # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # The 3rd is (1, 2, 2, 1).
    
    # Pattern for floor((S+1)/2):
    # The sequence is:
    # For i = 1 to N:
    # If i < (N+1)/2: place K copies of i at the end, in order of i.
    # If i > (N+1)/2: place K copies of i at the beginning, in reverse order of i.
    # If i == (N+1)/2: place K copies of i in the middle.
    # But the "middle" is tricky. Let's use the property:
    # The sequence is: 
    # [N//2] * K, then [N, N-1, ..., N//2 + 1] each K times, 
    # then [1, 2, ..., N//2 - 1] each K times.
    # Let's check Sample 1: N=2, K=2. N//2 = 1.
    # [1]*2, [2]*2, []. Result: 1 1 2 2. Wrong. Sample 1 is 1 2 2 1.
    
    # Let's re-evaluate: the middle sequence is the one that is 
    # lexicographically just smaller than its complement.
    # To be just smaller than its complement, it must start with the 
    # smallest possible value that allows it to be "large".
    # Actually, the pattern is:
    # The first K elements are (N+1)//2.
    # Then the remaining elements are arranged such that the sequence is 
    # the largest possible sequence that is still smaller than its complement.
    # That means for the remaining digits, we want them in descending order,
    # but we must ensure the overall sequence < complement.
    # If we start with (N+1)//2, the complement also starts with (N+1)//2.
    # To make the sequence as large as possible but < complement,
    # we want the remaining digits to be as large as possible, 
    # but the first digit that differs must be smaller than its complement.
    # The remaining digits are {1...N} \ {(N+1)//2}, each K times.
    # The largest such sequence is:
    # (N, N, ..., N), (N-1, ..., N-1), ..., (1, ..., 1).
    # But we must check if this is < complement.
    # Complement of ( (N+1)//2 * K, N*K, (N-1)*K, ..., 1*K )
    # is ( (N+1)//2 * K, 1*K, 2*K, ..., N*K ).
    # Since N > (N+1)//2, the first sequence is actually LARGER.
    # To make it smaller, we need to flip the order of the remaining.
    # The sequence is: (N+1)//2 * K, then 1*K, 2*K, ..., N*K (excluding (N+1)//2).
    # Wait, Sample 3: N=6, K=1. (N+1)//2 = 3.
    # Sequence: 3, 6, 5, 4, 2, 1.
    # This is: (N+1)//2, then N, N-1, ..., 1 (excluding (N+1)//2).
    # Let's check Sample 1: N=2, K=2. (N+1)//2 = 1.
    # Sequence: 1, 1, 2, 2. Still not 1 2 2 1.
    
    # Let's try: The middle sequence is the one where we place 
    # (N+1)//2 at the first and last positions? No.
    # Let's use the property: the sequence is the "reverse" of the 
    # lexicographically smallest sequence, but mirrored.
    # Smallest: 1*K, 2*K, ..., N*K.
    # Mirror: N*K, (N-1)*K, ..., 1*K.
    # The middle sequence is:
    # For i = 1 to N*K:
    # If i <= K: result[i] = (N+1)//2
    # If i > K: ...
    # Actually, the simplest pattern for floor((S+1)/2) is:
    # The sequence is: (N+1)//2 repeated K times, 
    # then (N, N-1, ..., 1) each K times, but skipping (N+1)//2,
    # AND the very last element is moved to the front? No.
    
    # Correct Pattern:
    # The sequence is: 
    # 1. Start with (N+1)//2 repeated K times.
    # 2. Then list all other numbers from N down to 1, each K times.
    # 3. BUT, to make it the floor((S+1)/2)-th, we need to adjust.
    # For N=2, K=2: (N+1)//2 = 1. Sequence: 1, 1, 2, 2. 
    # But the answer is 1, 2, 2, 1.
    # This means one '1' is at the start and one '1' is at the end.
    # The pattern is: 
    # Place (N+1)//2 at the first position, then 
    # place all other numbers in descending order, 
    # then place the remaining (K-1) copies of (N+1)//2 at the end.
    
    # Check Sample 1: N=2, K=2. (N+1)//2 = 1.
    # 1, (2, 2), (1). Result: 1 2 2 1. Correct!
    # Check Sample 3: N=6, K=1. (N+1)//2 = 3.
    # 3, (6, 5, 4, 2, 1), (). Result: 3 6 5 4 2 1. Correct!
    # Check Sample 4: N=3, K=3. (N+1)//2 = 2.
    # 2, (3, 3, 3, 1, 1, 1), (2, 2). Result: 2 3 3 3 1 1 1 2 2.
    # Sample 4 output: 2 2 2 1 3 3 3 1 1. 
    # My pattern is wrong. Let's look at Sample 4: 2 2 2 1 3 3 3 1 1.
    # It starts with 2 2 2 (which is (N+1)//2 repeated K times).
    # Then it has 1 3 3 3 1 1.
    # This is: 1, then 