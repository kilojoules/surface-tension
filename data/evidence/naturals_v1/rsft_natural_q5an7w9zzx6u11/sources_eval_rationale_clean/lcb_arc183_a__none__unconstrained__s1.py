The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional paradigm, emphasizing the transformation of data streams. In this specific problem, we need to determine the sequence at index $\lfloor (S+1)/2 \rfloor$. Since the set of all good sequences is symmetric (replacing each element $x$ with $N-x+1$ and reversing the sequence preserves the "good" property and maps the $i$-th sequence to the $(S-i+1)$-th sequence), the middle sequence is the one that is "self-dual" under the transformation $x \to N-x+1$ and reversal. Specifically, for the $\lfloor (S+1)/2 \rfloor$-th sequence, we can leverage the fact that the lexicographical middle of all permutations of a multiset is reached by filling the sequence from both ends inward, picking the "middle" available characters.

For this problem, the $\lfloor (S+1)/2 \rfloor$-th sequence is constructed by placing the available numbers in a specific balanced way: for each pair of positions $(i, NK-i+1)$, we place $x$ and $N-x+1$. To minimize the sequence lexicographically while staying at the median, we place the smaller of the pair at the earlier position and the larger at the later position, but we must distribute the values such that we don't "drift" too far from the center. The result is a sequence where we place $1, 2, \dots, N$ and their complements $N, N-1, \dots, 1$ symmetrically.

```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of sequences S is (NK)! / (K!)^N.
    # The sequence at index floor((S+1)/2) is the lexicographical median.
    # Due to the symmetry of the multiset, the median sequence is the one
    # that is "balanced". For every element x at position i, 
    # the element at position (NK - i + 1) is (N - x + 1).
    # To find the floor((S+1)/2)-th, we effectively want the sequence
    # that is as small as possible while remaining the "mirror" of itself
    # under the transformation x -> N-x+1.
    
    # This results in a pattern where we place 1s and Ns, then 2s and N-1s, etc.
    # Specifically, for the first K positions, we want to place the smallest 
    # possible values that allow the mirror to be valid.
    # The median sequence is constructed by:
    # For i from 1 to N/2:
    #   Place K copies of i at the earliest available slots
    #   Place K copies of (N-i+1) at the latest available slots
    # If N is odd, the middle element (N+1)//2 fills the remaining center slots.
    
    # We can construct this using a list and then joining it.
    # Since we cannot use loops, we use list comprehensions.
    
    # Create a list of size N*K
    # For N=3, K=3: 
    # i=1: indices 0,1,2 get '1', indices 6,7,8 get '3'
    # i=2: indices 3,4,5 get '2'
    # Result: 1 1 1 2 2 2 3 3 3 -> Wait, Sample 4 says 2 2 2 1 3 3 3 1 1
    # Let's re-evaluate. The sample 4 (N=3, K=3) output is 2 2 2 1 3 3 3 1 1.
    # This is the sequence where the middle value (N+1)//2 is placed first,
    # then the smaller values, then the larger values, then the smaller again.
    # Actually, the pattern for the median is:
    # Place (N+1)//2 for K times, then 1 for K times, then N for K times, 
    # then 2 for K times, then N-1 for K times...
    # But the sample 4 shows: 2 2 2 (mid) 1 3 3 3 (large) 1 1 (small)
    # Let's look closer: 2 2 2 | 1 3 3 3 1 1
    # The middle value (N+1)//2 is placed first. 
    # Then we alternate between the smallest and largest remaining.
    # For N=3, K=3: Mid=2. Sequence: [2]*3 + [1]*1 + [3]*3 + [1]*2
    # This is not quite right. Let's use the property:
    # The median sequence is the one where we place the values in the order:
    # (N+1)//2, 1, N, 2, N-1, 3, N-2... 
    # But we must distribute the K counts.
    # The correct logic for the median of these permutations is:
    # The sequence is S_i = (N+1)//2 if i is in the middle block,
    # otherwise it follows a mirrored pattern.
    # Actually, the simplest way to describe the median is:
    # For i = 1 to N*K:
    # If i <= K*(N//2), we are in the first half.
    # The sample 4: 2 2 2 1 3 3 3 1 1
    # Indices: 0 1 2 (val 2), 3 (val 1), 4 5 6 (val 3), 7 8 (val 1)
    # This looks like: 
    # 1. Middle value (N+1)//2 repeated K times.
    # 2. Then 1 repeated 1 time, then N repeated K times, then 1 repeated K-1 times.
    # 3. Then 2 repeated 1 time, then N-1 repeated K times, then 2 repeated K-1 times.
    # Wait, the sample 4 is 2 2 2 1 3 3 3 1 1.
    # That is: [2]*3, [1]*1, [3]*3, [1]*2.
    # Let's try: Mid, then 1, then N, then 2, then N-1...
    # For N=3, K=3: Mid=2. 
    # Sequence: 2,2,2, 1, 3,3,3, 1,1
    # This is: Mid(K), 1(1), N(K), 1(K-1), 2(1), N-1(K), 2(K-1)...
    # Let's check Sample 1: N=2, K=2. Mid=(2+1)//2 = 1.
    # Sequence: 1(2), 2(2), 1(0) -> 1 1 2 2. But sample 1 says 1 2 2 1.
    # Let's re-read: Sample 1 (2,2) -> 1 2 2 1.
    # Sample 4 (3,3) -> 2 2 2 1 3 3 3 1 1.
    # In both, the sequence is a palindrome if you replace x with N-x+1.
    # Sample 1: 1 2 2 1 -> mirror is 2 1 1 2. Not a palindrome.
    # Wait, the mirror of 1 2 2 1 is: reverse(1 2 2 1) = 1 2 2 1, then replace x with 3-x: 2 1 1 2.
    # The median sequence S is the one where S = mirror(S).
    # S_i = N + 1 - S_{NK-i+1}.
    # For Sample 1: S_1=1, S_4=2; S_2=2, S_3=1. Sequence: 1 2 1 2? No, 1 2 2 1.
    # Check: S_1=1, S_4=2 (1=3-2), S_2=2, S_3=1 (2=3-1). Correct.
    # For Sample 4: S_1=2, S_9=1 (2=4-1), S_2=2, S_8=1 (2=4-1), S_3=2, S_7=1 (2=4-1),
    # S_4=1, S_6=3 (1=4-3), S_5=3 (3=4-1? No).
    # Let's re-calculate mirror for Sample 4: 2 2 2 1 3 3 3 1 1
    # Indices: 1 2 3 4 5 6 7 8 9
    # S_1=2, S_9=1. 2+1=3. (N+1=4). Not mirror.
    # Let's use the property: the median is the sequence that is lexicographically
    # the middle. This is achieved by picking the "middle" available character
    # for the first position, then the "middle" for the second, and so on.
    # For a multiset, the "middle" character is the one that splits the 
    # remaining permutations into two equal halves.
    # This happens when we pick character 'c' such that the number of permutations
    # starting with characters < c is equal to the number of permutations
    # starting with characters > c.
    # This is solved by: for each position i, pick the smallest c such that
    # count(chars < c) * perms(remaining) >= (total_perms_at_i + 1) // 2.
    # Since we need the median of ALL, we can just maintain the counts of each
    # number andに pick the character that balances the remaining permutations.
    # For the median, we simply need to pick the character c that is the 
    # median of the current multiset of available characters.
    # The median of a multiset is the element at index (total_size - 1) // 2.
    
    # Let's test this hypothesis:
    # Sample 1: N=2, K=2. Multiset: {1,1,2,2}
    # i=1: median is index (4-1)//2 = 1. Sorted: 1,1,2,2. Index 1 is '1'.
    # Remaining: {1,2,2}. i=2: median index (3-1)//2 = 1. Sorted: 1,2,2. Index 1 is '2'.
    # Remaining: {1,2}. i=3: median index (2-1)//2 = 0. Sorted: 1,2. Index 0 is '1'.
    # Wait, this is not matching. Let's try index total_size // 2.
    # Sample 1: {1,1,2,2} -> idx 2 is '2'. {1,1,2} -> idx 1 is '1'. {1,2} -> idx 1 is '2'. {1} -> idx 0 is '1'.
    # Result: 2 1 2 1. Still not 1 2 2 1.
    
    # Correct logic for median of permutations:
    # The median sequence is the one where we always pick the character c
    # such that the number of permutations starting with characters < c
    # is just under half of the total.
    # Total permutations = (NK)! / (K!)^N.
    # Permutations starting with char c = (NK-1)! / (K!^{N-1} * (K-1)!)
    # = Total * K / (NK).
    # We want sum_{j=1}^{c-1} (Total * K / NK) < Total / 2
    # (c-1) * K / NK < 1/2  =>  c-1 < N/2  =>  c < N/2 + 1.
    # So c = floor(N/2) + 1.
    # This means for the first position, we pick c = (N // 2) + 1.
    # Then we update the counts and repeat.
    # For Sample 1: N=2, K=2. c = (2//2)+1 = 2. First char is 2.
    # Remaining: {1,1,2}. Total perms = 3!/2!1! = 3.
    # c=1: perms = 2!/1!1! = 2. 2 >= (3+1)//2. So c=1.
    # Remaining: {1,2}. Total perms = 2!/1!1