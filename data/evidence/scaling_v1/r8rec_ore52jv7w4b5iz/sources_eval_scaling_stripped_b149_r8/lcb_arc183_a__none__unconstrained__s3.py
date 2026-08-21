```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence 
    # (a_1, a_2, ..., a_{NK}) is the i-th sequence, then the sequence
    # (N+1-a_1, N+1-a_2, ..., N+1-a_{NK}) is the (S - i + 1)-th sequence.
    # The middle sequence is the one where the sequence is "balanced".
    # Specifically, for the floor((S+1)/2)-th sequence, we want the 
    # lexicographically smallest sequence that is "greater than or equal to"
    # its symmetric counterpart.
    
    # However, a simpler observation for the "middle" of all permutations 
    # of a multiset is that we can determine the elements one by one.
    # For the first position, if we have N choices, the first floor(S/N) 
    # sequences start with 1, etc.
    # But since we need exactly the middle, we can use the property that
    # the middle sequence of all permutations of {1*K, 2*K, ..., N*K}
    # starts with the value ceil(N/2).
    
    # Let's refine: the middle sequence is the one that "balances" the 
    # distribution of numbers. 
    # For N=2, K=2: sequences are (1,1,2,2), (1,2,1,2), (1,2,2,1), 
    # (2,1,1,2), (2,1,2,1), (2,2,1,1). S=6. floor(7/2)=3. Result: (1,2,2,1).
    # For N=6, K=1: S=720. floor(721/2)=360. Result: (3, 6, 5, 4, 2, 1).
    
    # The pattern for the floor((S+1)/2)-th sequence is:
    # 1. The first element is (N+1)//2.
    # 2. The remaining elements are filled such that we use the 
    #    remaining counts of numbers.
    # 3. To stay just below the exact middle, we fill the remaining 
    #    slots by placing the largest available numbers first, 
    #    then the smallest, in a specific mirrored priority.
    
    # Correct logic for the middle sequence of a multiset:
    # The sequence is constructed by:
    # - First element: (N + 1) // 2
    # - Then, for the remaining positions, we want to be as "large" as 
    #   possible without crossing the halfway point of the total 
    #   permutations.
    # This results in:
    # - All remaining instances of numbers > (N+1)//2 in descending order.
    # - All remaining instances of numbers < (N+1)//2 in descending order.
    # - All remaining instances of (N+1)//2 in descending order (which is just the value).
    # Wait, the sample 4 (N=3, K=3) gives 2 2 2 1 3 3 3 1 1.
    # Let's analyze Sample 4: N=3, K=3. Mid is 2.
    # Sequence: 2, 2, 2, 1, 3, 3, 3, 1, 1.
    # This looks like: 
    # 1. All K copies of (N+1)//2.
    # 2. All K copies of (N+1)//2 - 1 (descending).
    # 3. All K copies of (N+1)//2 + 1 (descending).
    # 4. All K copies of (N+1)//2 - 2...
    # Actually, the pattern is:
    # Start with (N+1)//2.
    # Then use all remaining K-1 of (N+1)//2.
    # Then alternate between the remaining numbers from the middle outwards.
    # For N=3, K=3: Mid=2. 
    # Sequence: 2 (K times), then 1 (K times), then 3 (K times), then 1 (remaining)... 
    # No, that's not it.
    
    # Let's re-evaluate: The middle sequence is the one that is 
    # "complementary" to itself.
    # The sequence is:
    # For i from 1 to N:
    # If i < (N+1)//2: place K copies of i at the end.
    # If i > (N+1)//2: place K copies of i in the middle.
    # If i == (N+1)//2: place K copies of i at the start.
    # For N=3, K=3: 
    # i=2: 2 2 2 (start)
    # i=3: 3 3 3 (middle)
    # i=1: 1 1 1 (end)
    # Result: 2 2 2 3 3 3 1 1 1. 
    # Sample 4 says: 2 2 2 1 3 3 3 1 1. This is different.
    
    # Let's look at Sample 3: N=6, K=1. Mid=3.
    # Output: 3 6 5 4 2 1.
    # This is: Mid, then all numbers > Mid descending, then all numbers < Mid descending.
    # For N=3, K=3: Mid=2.
    # Output: 2 2 2, then 3 3 3 (descending), then 1 1 1 (descending).
    # Wait, Sample 4 output is 2 2 2 1 3 3 3 1 1. 
    # Let me re-read Sample 4: 2 2 2 1 3 3 3 1 1.
    # That is: 2(3x), 1(1x), 3(3x), 1(2x).
    # This means the "middle" is reached by filling the 
    # most significant digits with the middle value, then 
    # the smallest possible value that allows the rest to be 
    # the "largest" possible.
    
    # Correct Logic:
    # To find the floor((S+1)/2)-th sequence:
    # The first digit is (N+1)//2.
    # Then we need to find the floor((S_remaining + 1)/2)-th sequence 
    # of the remaining multiset.
    # This is a recursive property.
    # For a multiset, the middle sequence is:
    # 1. Use one instance of the median value.
    # 2. Then, to stay in the first half of the total permutations,
    #    we want the remaining sequence to be the "largest" possible 
    #    sequence that is still <= the middle.
    # The middle of all permutations is the sequence that, when 
    # mirrored (x -> N+1-x), is its own reverse.
    # The sequence is:
    # For j from 0 to K-1:
    #   Place (N+1)//2
    #   Place all numbers > (N+1)//2 in descending order (K times each)
    #   Place all numbers < (N+1)//2 in descending order (K times each)
    # But we only do the "all numbers" part once.
    
    # Let's try this:
    # 1. K copies of (N+1)//2
    # 2. K copies of (N+1)//2 - 1, K copies of (N+1)//2 + 1, ...
    # Actually, the simplest pattern that fits all samples:
    # The sequence is:
    # (N+1)//2 repeated K times,
    # then (N+1)//2 - 1 repeated 1 time,
    # then (N+1)//2 + 1 repeated K times,
    # then (N+1)//2 - 1 repeated K-1 times,
    # then (N+1)//2 + 2 repeated K times...
    # No.
    
    # Let's use the property: the middle sequence is the one that 
    # starts with the median, and then the remaining is the 
    # "largest" sequence that is still in the first half.
    # That means:
    # 1. First element: (N+1)//2
    # 2. Remaining: 
    #    - All elements < (N+1)//2 in descending order, 
    #      but we must place one instance of the smallest 
    #      available element to "keep" it in the first half, 
    #      then all larger elements descending, then the rest.
    # Actually, the pattern is:
    # - K copies of (N+1)//2
    # - 1 copy of (N+1)//2 - 1
    # - K copies of N, K copies of N-1 ... K copies of (N+1)//2 + 1
    # - K-1 copies of (N+1)//2 - 1, K copies of (N+1)//2 - 2 ...
    
    # Let's test N=3, K=3:
    # 1. 2 2 2
    # 2. 1 (one copy)
    # 3. 3 3 3 (descending from N down to Mid+1)
    # 4. 1 1 (remaining copies of Mid-1 down to 1)
    # Result: 2 2 2 1 3 3 3 1 1. MATCHES SAMPLE 4!
    
    # Let's test N=6, K=1:
    # 1. 3 (one copy)
    # 2. 2 (one copy) - wait, (N+1)//2 is 3. Mid-1 is 2.
    # 3. 6 5 4 (descending from 6 down to 4)
    # 4. 2 (remaining copies of 2 down to 1) - wait, 2 was already used.
    # If K=1, the "1 copy of Mid-1" and "remaining copies" 
    # just means the number Mid-1 appears once.
    # Result: 3 2 6 5 4 1. Sample 3 says 3 6 5 4 2 1.
    # My rule 2 and 3 are swapped.
    
    # Corrected Rule:
    # 1. K copies of (N+1)//2
    # 2. K copies of N, N-1, ..., (N+1)//2 + 1 (each K times)
    # 3. 1 copy of (N+1)//2 - 1
    # 4. K-1 copies of (N+1)//2 - 1, then K copies of (N+1)//2 - 2, ..., 1 (each K times)
    # Wait, if K=1, rule 3 is 1 copy of 2, rule 4 is 0 copies of 2, then 1 copy of 1.
    # Result for N=6, K=1: 3, 6, 5, 4, 2, 1. MATCHES SAMPLE 3!
    # Result for N=3, K=3: 2 2 2, 3 3 3, 1, 1 1. 
    # Sample 4: 2 2 2 1 3 3 3 1 1.
    # My rule 2 and 3 are still swapped for Sample 4.
    
    # Let's look at Sample 4 again: 2 2 2 1 3 3 3 1 1.
    # It is: Mid(K), Mid-1(1), Mid+1(K), Mid-1(K-1).
    # This means:
    # 1. Mid (K times)
    # 2. Mid-1 (1 time)
    # 3. Mid+1...N (each K times, descending)
    # 4. Mid-1 (K-1 times), Mid-2...1 (each K times, descending)
    # Let's check N=6, K=1:
    # 1. 3 (1 time)
    # 2. 2 (1 time)
    