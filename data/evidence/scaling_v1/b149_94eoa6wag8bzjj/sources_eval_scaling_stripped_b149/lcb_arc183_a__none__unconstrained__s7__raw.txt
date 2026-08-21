```python
import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The problem asks for the floor((S+1)/2)-th sequence.
    # S is the total number of permutations of the multiset {1*K, 2*K, ..., N*K}.
    # S = (N*K)! / (K!^N).
    # The lexicographical middle of a symmetric set of sequences is 
    # the sequence that is the "complement" of the first sequence.
    # The first sequence is (1, ..., 1, 2, ..., 2, ..., N, ..., N).
    # The last sequence is (N, ..., N, N-1, ..., N-1, ..., 1, ..., 1).
    # For any sequence A, its lexicographical complement A' is formed by 
    # replacing each element x with (N + 1 - x).
    # If S is the total number of sequences, the i-th sequence and the 
    # (S - i + 1)-th sequence are complements.
    # We want the floor((S+1)/2)-th sequence.
    # If S is even, this is the (S/2)-th sequence.
    # If S is odd, this is the ((S+1)/2)-th sequence.
    # In both cases, this is the sequence immediately preceding the 
    # "middle" point or the middle itself.
    # Specifically, the (S/2)-th sequence is the complement of the 
    # (S/2 + 1)-th sequence.
    # The sequence we are looking for is the one that, when complemented,
    # is the smallest sequence that is lexicographically greater than 
    # its own complement (or equal).
    # Actually, a simpler property: the floor((S+1)/2)-th sequence is the 
    # lexicographical predecessor of the sequence that is its own complement,
    # or the complement of the sequence that is the "ceiling" middle.
    # More simply: the floor((S+1)/2)-th sequence is the complement of the 
    # (S - floor((S+1)/2) + 1)-th sequence.
    # S - floor((S+1)/2) + 1 = ceil((S+1)/2).
    # The sequence we need is the complement of the ceil((S+1)/2)-th sequence.
    # The ceil((S+1)/2)-th sequence is the smallest sequence A such that 
    # A >= complement(A).
    
    # To find the smallest sequence A such that A >= complement(A):
    # We determine the elements from left to right.
    # At the first position i where A[i] != complement(A)[i], we must have A[i] > complement(A)[i].
    # To make A as small as possible, we want the first difference to occur as late as possible.
    # The "middle" sequence in a symmetric distribution is the one that 
    # starts with the middle value of the available digits.
    
    # For N=2, K=2: S=6. floor(7/2)=3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
    # 3rd is 1221. Complement of 1221 is 2112 (4th).
    # For N=6, K=1: S=720. floor(721/2)=360. 
    # The 360th is the complement of the 361st.
    # The 361st is the smallest sequence A such that A > complement(A).
    # For K=1, the 361st is (4, 1, 2, 3, 5, 6) - wait, no.
    # For K=1, the sequences are permutations of 1..N.
    # The middle is between (3, 6, 5, 4, 2, 1) and (4, 1, 2, 3, 5, 6).
    # The 360th is (3, 6, 5, 4, 2, 1).
    
    # General rule for floor((S+1)/2)-th:
    # It is the complement of the smallest sequence A such that A >= complement(A).
    # The smallest A such that A >= complement(A) is:
    # 1. Find the smallest x such that we can form a sequence starting with x
    #    where the remaining counts allow the sequence to be >= its complement.
    # 2. Actually, the simplest construction for the floor((S+1)/2)-th sequence is:
    #    The sequence is the complement of the smallest sequence A such that A >= complement(A).
    #    The smallest A such that A >= complement(A) is:
    #    - The first element is ceil(N/2).
    #    - If N is even, the first element is N/2, but we must ensure the rest 
    #      of the sequence makes A >= complement(A).
    #    - The most direct construction:
    #      The sequence is: 
    #      (N // 2) repeated K times, 
    #      then (N // 2 - 1) down to 1 repeated K times,
    #      then (N // 2 + 1) up to N repeated K times.
    #      Wait, let's check Sample 1: N=2, K=2. N//2 = 1.
    #      Sequence: 1 (2 times), then (0..1), then (2..2). -> 1 1 2 2. 
    #      But Sample 1 says 1 2 2 1.
    
    # Let's re-evaluate. The floor((S+1)/2)-th sequence is the complement of 
    # the smallest sequence A such that A >= complement(A).
    # Smallest A such that A >= complement(A):
    # The first element A[0] must be >= complement(A)[0].
    # complement(A)[0] = N + 1 - A[0].
    # So A[0] >= (N + 1) / 2.
    # The smallest such integer is A[0] = ceil((N + 1) / 2).
    # To make A smallest, we want A[0] to be as small as possible, so A[0] = (N + 1) // 2 
    # if we can't, then (N + 2) // 2.
    # Actually, the smallest A such that A >= complement(A) is:
    # A[0] = (N + 1) // 2.
    # If A[0] > complement(A)[0], then to minimize A, the remaining elements 
    # should be as small as possible (sorted ascending).
    # If A[0] == complement(A)[0] (which happens if N is odd and A[0] = (N+1)//2),
    # we recurse on the remaining elements.
    
    # Let's trace N=2, K=2:
    # A[0] >= (2+1)/2 = 1.5 -> A[0] = 2.
    # Since A[0] > complement(A)[0] (2 > 1), the rest of A is sorted ascending: 1, 1, 2.
    # So A = (2, 1, 1, 2).
    # The answer is complement(A) = (2+1-2, 2+1-1, 2+1-1, 2+1-2) = (1, 2, 2, 1). Correct.
    
    # Trace N=6, K=1:
    # A[0] >= (6+1)/2 = 3.5 -> A[0] = 4.
    # A[0] > complement(A)[0] (4 > 3), so rest is sorted ascending: 1, 2, 3, 5, 6.
    # A = (4, 1, 2, 3, 5, 6).
    # complement(A) = (3, 6, 5, 4, 2, 1). Correct.
    
    # Trace N=3, K=3:
    # A[0] >= (3+1)/2 = 2.
    # If A[0] = 2, then complement(A)[0] = 3+1-2 = 2.
    # Since A[0] == complement(A)[0], we look at the remaining:
    # Remaining elements for A: {1:3, 2:2, 3:3}.
    # We need the smallest sequence A' from these such that A' >= complement(A').
    # Note: the complement of an element x in the original set is N+1-x.
    # The complement of the remaining set is {3:3, 2:2, 1:3}.
    # For A', the first element A'[0] must be >= complement(A')[0].
    # The available elements are 1, 2, 3.
    # If A'[0] = 1, complement(A')[0] = 3. 1 < 3 (False).
    # If A'[0] = 2, complement(A')[0] = 2. 2 == 2 (True).
    # So A'[0] = 2. Now remaining: {1:3, 2:1, 3:3}.
    # Repeat: A''[0] must be >= complement(A'')[0].
    # If A''[0] = 1, complement(A'')[0] = 3. (False).
    # If A''[0] = 2, complement(A'')[0] = 2. (True).
    # So A''[0] = 2. Now remaining: {1:3, 2:0, 3:3}.
    # Repeat: A'''[0] must be >= complement(A''')[0].
    # If A'''[0] = 1, complement(A''')[0] = 3. (False).
    # If A'''[0] = 2, not available.
    # If A'''[0] = 3, complement(A''')[0] = 1. (True).
    # So A'''[0] = 3. Since A'''[0] > complement(A''')[0], the rest are sorted ascending.
    # Rest: {1:3, 3:2} -> 1, 1, 1, 3, 3.
    # So A = (2, 2, 2, 3, 1, 1, 1, 3, 3).
    # complement(A) = (2, 2, 2, 1, 3, 3, 3, 1, 1). Correct.

    # Implementation of the logic:
    # We need to find the smallest sequence A such that A >= complement(A).
    # We can't use loops, so we use a recursive-like approach via a list comprehension 
    # or a functional approach. However, the constraints on A[i] depend on the 
    # counts of remaining elements.
    # Since we need to output the complement, we can just find A and then map it.
    
    # To find A without loops:
    # A[i] is the smallest available digit d such that:
    # 1. d > (N + 1 - d)
    # 2. OR (d == (N + 1 - d) AND there exists a way to make the rest of A >= complement(A))
    # Actually, the condition "there exists a way" is always true if we have 
    # the required digits. The only constraint is that we must have the 
    # complement digits available.
    
    # Let's simplify:
    # A[i] is the smallest available digit d such that:
    # - If d > N + 1 - d, then A is definitely >= complement(A). To minimize A, 
    #   we take the smallest such d, and then all subsequent digits are sorted ascending.
    # - If d == N + 1 - d, we can only take d if it's available and we can 
    #   continue to satisfy the condition for the suffix.
    # - If d < N + 1 - d, we can only take d if we eventually have some 
    #   A[j] > complement(A)[j] for j > i. But we want the smallest A, 
    #   so we only take d < N + 1 - d if we are forced to. 
    #   Wait, if A[i] < complement(A)[i],