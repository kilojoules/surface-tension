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
    # Due to the symmetry of the set of all good sequences, 
    # if a sequence A is the i-th, then its "complement" 
    # (where each value x is replaced by N - x + 1) is the (S - i + 1)-th.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, we want the sequence that is lexicographically 
    # just before or equal to its complement.
    
    # To find the floor((S+1)/2)-th sequence, we can use a greedy approach.
    # At each position, we try to place the smallest available number.
    # We calculate how many sequences start with the current prefix.
    # If the number of sequences starting with the current prefix is 
    # less than the target rank, we subtract that count from the rank 
    # and try the next available number.
    
    # However, calculating S and the rank explicitly is impossible with 
    # standard floats and slow with large integers.
    # Observation: The floor((S+1)/2)-th sequence is the one that 
    # "balances" the distribution of numbers.
    # For a given N and K, the middle sequence is constructed by:
    # 1. Filling the first half of the slots with numbers in a way that 
    #    we stay just under the 50% mark of total permutations.
    # 2. A known property for this specific problem (finding the median 
    #    of multiset permutations) is that we can determine the elements 
    #    by comparing the remaining counts.
    
    # Let's use the property: the middle sequence is the one where we 
    # try to place the smallest possible number such that the number 
    # of permutations remaining is at least half of the total remaining.
    
    # Since we cannot compute the total S, we use the fact that 
    # the middle sequence is the one that is "lexicographically 
    # equivalent" to the average of the first and last sequence.
    # The first is (1*K, 2*K, ..., N*K)
    # The last is (N*K, (N-1)*K, ..., 1*K)
    # The middle sequence is ( (1+N)/2 * K, ... )
    
    # More accurately: for each position i from 0 to NK-1:
    # We want to pick the smallest x from 1 to N (if count[x] > 0)
    # such that the number of permutations starting with (prefix, x)
    # is >= the remaining rank.
    
    # But we can't compute the number of permutations.
    # Wait, the problem asks for the floor((S+1)/2)-th.
    # For N=2, K=2: S=6. Rank = floor(7/2) = 3.
    # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # 3rd is (1,2,2,1).
    
    # Notice: (1,2,2,1) is the "reverse" of (1,2,2,1) if we map 1->2 and 2->1.
    # The middle sequence is the one that is "self-complementary" 
    # in terms of rank.
    
    # Correct logic for the median of multiset permutations:
    # We can determine the elements one by one. 
    # For the current position, we check if the number of permutations 
    # starting with 1, 2, ..., x-1 is less than S/2.
    # This is equivalent to checking if the "average" sequence 
    # has x at this position.
    
    # The middle sequence is simply the sequence where we place 
    # the numbers in a specific balanced order.
    # For N=2, K=2, it's 1, 2, 2, 1.
    # For N=6, K=1, it's 3, 6, 5, 4, 2, 1.
    # For N=3, K=3, it's 2, 2, 2, 1, 3, 3, 3, 1, 1.
    
    # Pattern: 
    # If N is even, the middle sequence starts with N/2, then 
    # follows a pattern of placing the remaining numbers.
    # Actually, the middle sequence is the one that is 
    # "lexicographically" in the center.
    # It is known that the median of all permutations of a multiset 
    # is the sequence where we place the elements in the order:
    # For i = 0 to NK-1:
    # The element is the smallest x such that the number of 
    # permutations of the remaining elements is >= the remaining rank.
    
    # Since we can't use loops, we use a mathematical construction.
    # The middle sequence is:
    # For N=3, K=3: 2 2 2 1 3 3 3 1 1
    # This looks like: 
    # - All K copies of (N+1)//2
    # - All K copies of 1, then all K copies of N, 
    #   then all K copies of 2, then all K copies of N-1...
    # Wait, the sample 3 (N=6, K=1) is 3 6 5 4 2 1.
    # Let's analyze N=6, K=1: 3, 6, 5, 4, 2, 1.
    # The middle of (1,2,3,4,5,6) and (6,5,4,3,2,1) is 3.5.
    # The sequence is: 3, 6, 5, 4, 2, 1.
    # This is: (N//2), then (N, N-1, ..., N//2 + 1) reversed? No.
    # It's: N//2, then N, N-1, ..., N//2 + 1, then N//2 - 1, ..., 1.
    # Let's check N=3, K=3: N//2 = 1. 
    # Sequence: 1(3), 3(3), 2(3) -> 1 1 1 3 3 3 2 2 2.
    # But sample says: 2 2 2 1 3 3 3 1 1.
    # That is: (N+1)//2 repeated K times, then 1 repeated K times, 
    # then N repeated K times, then 2 repeated K times, then N-1...
    
    # Let's re-evaluate:
    # For N=2, K=2: (N+1)//2 = 1. 
    # 1(2), 2(2) -> 1 1 2 2. But sample says 1 2 2 1.
    # Wait, the sample 1 (N=2, K=2) is 1 2 2 1.
    # That is: 1, 2, 2, 1.
    # This is: 1, (2 repeated K times), 1.
    
    # Let's look at the symmetry. The middle sequence S_mid 
    # must satisfy S_mid = complement(S_mid) if S is odd, 
    # or be the one just before the complement if S is even.
    # The complement of (S_1, ..., S_{NK}) is (N-S_1+1, ..., N-S_{NK}+1).
    # For N=2, K=2: (1, 2, 2, 1) -> complement is (2, 1, 1, 2).
    # (1, 2, 2, 1) is indeed the 3rd of 6.
    
    # The construction for the middle sequence is:
    # For i = 1 to N:
    # If i < (N+1)/2: place i at the end and N-i+1 at the beginning?
    # No, the order is:
    # For the middle sequence, we want to place the "middle" value first.
    # If N is odd, the middle value is (N+1)//2.
    # If N is even, the middle values are N//2 and N//2 + 1.
    
    # Correct Pattern:
    # The sequence is constructed by placing the values in the order:
    # (N+1)//2 (K times), 
    # then 1 (K times), N (K times), 
    # then 2 (K times), N-1 (K times)...
    # Until all numbers are used.
    # For N=2, K=2: (2+1)//2 = 1. 
    # 1(2), then 2(2). But the 1s are split? 
    # 1, 2, 2, 1.
    # This is: 1, 2(2), 1.
    # For N=3, K=3: (3+1)//2 = 2.
    # 2(3), 1(3), 3(3).
    # For N=6, K=1: (6+1)//2 = 3.
    # 3(1), 1(1), 6(1), 2(1), 5(1), 3(1)... no.
    # Let's try: 3, 6, 5, 4, 2, 1.
    # This is: 3, then (6, 5, 4), then (2, 1).
    
    # The general rule for the middle sequence:
    # 1. Start with the middle element M = (N+1)//2.
    # 2. Then list all elements > M in descending order.
    # 3. Then list all elements < M in descending order.
    # Each element is repeated K times.
    # However, the M element is split: one K at the start, 
    # but if N is even, the "middle" is between N/2 and N/2+1.
    
    # Let's refine:
    # If N is odd: M = (N+1)//2. 
    # Sequence: M(K), N(K), N-1(K), ..., M+1(K), M-1(K), ..., 1(K).
    # If N is even: M = N//2.
    # Sequence: M(K), N(K), N-1(K), ..., M+1(K), M-1(K), ..., 1(K).
    # Wait, for N=2, K=2: M=1. 1(2), 2(2). But sample is 1 2 2 1.
    # The only way to get 1 2 2 1 is if the first M is 1, then 2(2), then 1.
    # That means the K copies of M are split: one at the start, one at the end.
    # But the problem says K is the number of times each integer appears.
    # In 1 2 2 1, the number 1 appears twice, and 2 appears twice.
    # So the M(K) is split into 1 at the start and K-1 at the end?
    # No, that's not symmetric.
    
    # Let's use the property: the middle sequence is the one that 
    # is lexicographically the "average" of the first and last.
    # First: 1(K), 2(K), ..., N(K)
    # Last: N(K), N-1(K), ..., 1(K)
    # The middle sequence is:
    # For i = 1 to N:
    # If i is the middle element, it's placed based on the remaining.
    # Actually, the simplest construction is:
    # The sequence is: (N+1)//2 repeated K times, 
    # then N, N-1, ..., (N+1)//2 + 1 (each K times),
    # then (N+1)//2 - 1, ..., 1 (each K times).
    # BUT, if