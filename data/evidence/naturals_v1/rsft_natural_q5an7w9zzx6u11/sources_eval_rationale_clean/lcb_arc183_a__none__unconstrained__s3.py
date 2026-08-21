The constraint to avoid explicit loops forces a transition from imperative programming to a functional paradigm, emphasizing data transformation over state mutation. In this specific problem, since we need to determine the sequence element-by-element based on the number of permutations remaining, I will use `functools.reduce` to simulate the stateful construction of the sequence and list comprehensions for calculations.

To find the $\lfloor(S+1)/2\rfloor$-th sequence, we are essentially looking for the "middle" sequence. A key observation is that if a sequence $A$ is the $i$-th lexicographical sequence, its "complement" sequence $A'$ (where each element $x$ is replaced by $N - x + 1$) is the $(S - i + 1)$-th sequence. The middle sequence is the one that is "self-complementary" in terms of its rank. Specifically, the $\lfloor(S+1)/2\rfloor$-th sequence is the one that is lexicographically just smaller than or equal to its own complement.

However, a simpler approach to find the middle sequence is to realize that the middle sequence is the one that is "balanced" around the center of the lexicographical range. For any sequence $A$, its complement $A^c$ (replacing $x$ with $N-x+1$) satisfies $A < A^c$ if and only if at the first index $i$ where they differ, $A_i < N - A_i + 1$. The middle sequence is the lexicographically smallest sequence $A$ such that $A \geq A^c$ is false, or more simply, the sequence that is the "mirror" of the lexicographical median.

Actually, the most direct way to find the $\lfloor(S+1)/2\rfloor$-th sequence is to realize it is the sequence that is "halfway" through the permutations. For a set of multiset permutations, the sequence at rank $(S+1)/2$ is the one that is "symmetrical" to the lexicographical middle. The simplest construction for the middle sequence is to place the numbers in a way that they are balanced: for $N$ symbols, the middle sequence is the one that starts with the middle symbol $\lceil N/2 \rceil$ and arranges the others symmetrically.

Wait, the property is simpler: the $\lfloor(S+1)/2\rfloor$-th sequence is the one that is lexicographically the "median". For multiset permutations, the median sequence is the one where we use the symbols in a balanced fashion. Specifically, it is the sequence formed by taking the symbols $1, \dots, N$ and arranging them such that we use them in the order: $\lceil N/2 \rceil, \dots, 1, N, \dots, \lceil N/2 \rceil + 1$ repeated $K$ times? No.

Correct logic: The sequence at rank $(S+1)/2$ is the one that is "self-complementary" in the sense that if you replace $x$ with $N-x+1$, you get the sequence at rank $S - \text{rank}(A) + 1$. The sequence that satisfies $\text{rank}(A) = S - \text{rank}(A) + 1$ is the one where $A_i + A_{NK-i+1} = N+1$. For the middle sequence, we can simply output the sequence: $\lceil N/2 \rceil$ repeated $K$ times, then $\lceil N/2 \rceil - 1$ repeated $K$ times... down to 1, then $\lceil N/2 \rceil + 1$ repeated $K$ times... up to $N$. But the order must be lexicographical.

The middle sequence is actually: $\lceil N/2 \rceil$ repeated $K$ times, then $\lceil N/2 \rceil - 1$ repeated $K$ times... then $\lceil N/2 \rceil + 1$ repeated $K$ times... No.
Actually, the middle sequence is simply the sequence where we list the numbers from $\lceil N/2 \rceil$ down to 1, then $\lceil N/2 \rceil + 1$ up to $N$, each repeated $K$ times. Let's check Sample 1: $N=2, K=2$. $\lceil 2/2 \rceil = 1$. Sequence: $1(2), 2(2) \to 1, 1, 2, 2$. Wait, Sample 1 says $1, 2, 2, 1$.
Sample 3: $N=6, K=1$. $\lceil 6/2 \rceil = 3$. Sequence: $3, 2, 1, 4, 5, 6$. Sample 3 says $3, 6, 5, 4, 2, 1$.

The pattern is: the middle sequence is the one that starts with $\lceil N/2 \rceil$, then follows with the remaining numbers in **descending** order.
For $N=6, K=1$: $3, 6, 5, 4, 2, 1$.
For $N=2, K=2$: $1, 2, 2, 1$. (Starts with 1, then 2, 2, 1).
The rule is: The first element is $\lceil N/2 \rceil$. Then the remaining elements are filled by placing the largest available numbers as late as possible and smallest as early as possible? No.
The rule is: The sequence is $A_i$ such that $A_i + A_{NK-i+1} = N+1$. To be the $\lfloor(S+1)/2\rfloor$-th, we want the lexicographically smallest sequence such that $A \ge A^c$. This is achieved by setting $A_1 = \lceil (N+1)/2 \rceil$, and for $i > 1$, $A_i$ is as small as possible, which forces $A_{NK-i+1}$ to be as large as possible.

```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The middle sequence (floor((S+1)/2)-th) is the lexicographically smallest 
    # sequence A such that A >= A_complement, where A_complement is 
    # the sequence obtained by replacing x with N - x + 1.
    # This is achieved by:
    # 1. The first element is ceil((N+1)/2).
    # 2. For the remaining positions, we want the sequence to be as small as possible.
    # 3. However, we must maintain the property A_i + A_{NK-i+1} = N+1 to stay at the center.
    # 4. To make A lexicographically smallest while A >= A_complement:
    #    - A[0] = (N + 1) // 2 (if N is odd) or N // 2 (if N is even)
    #    - Actually, the simplest construction for the median is:
    #      The first element is (N + 1) // 2.
    #      Then we place all remaining 1s, then 2s... up to (N+1)//2 - 1.
    #      Then we place the remaining (N+1)//2's.
    #      Then we place the largest numbers in descending order to satisfy the complement.
    
    # Correct logic for the median of multiset permutations:
    # The sequence is: 
    # First, (N+1)//2 repeated K times.
    # Then, 1 repeated K times, 2 repeated K times ... ((N+1)//2 - 1) repeated K times.
    # Then, N repeated K times, N-1 repeated K times ... ((N+1)//2 + 1) repeated K times.
    # Wait, let's check Sample 1: N=2, K=2. (2+1)//2 = 1.
    # 1 repeated 2 times, then 2 repeated 2 times. -> 1 1 2 2. 
    # But Sample 1 says 1 2 2 1.
    
    # Let's re-evaluate: A is the median if A = A_complement.
    # If N=2, K=2, sequences are:
    # 1122, 1212, 1221, 2112, 2121, 2211.
    # The 3rd is 1221. Note: 1221 is the complement of 2112? No.
    # Complement of 1221 is 2112.
    # The 3rd and 4th are 1221 and 2112. floor((6+1)/2) = 3.
    # For N=6, K=1: 3 6 5 4 2 1.
    # This is: mid, then N, N-1, ... mid+1, mid-1, ... 1.
    
    mid = (N + 1) // 2
    # The pattern observed:
    # First element: mid
    # Then: N, N-1, ..., mid+1 (each K times)
    # Then: mid-1, mid-2, ..., 1 (each K times)
    # Then: mid (remaining K-1 times)
    # Let's check Sample 1: N=2, K=2. mid=1.
    # 1, (2 repeated 2 times), (none), (1 repeated 1 time) -> 1 2 2 1. Correct.
    # Sample 3: N=6, K=1. mid=3.
    # 3, (6, 5, 4), (2, 1), (none) -> 3 6 5 4 2 1. Correct.
    # Sample 4: N=3, K=3. mid=2.
    # 2, (3 repeated 3 times), (1 repeated 3 times), (2 repeated 2 times)
    # -> 2 3 3 3 1 1 1 2 2. 
    # Sample 4 output: 2 2 2 1 3 3 3 1 1. 
    # Wait, my pattern is wrong. Let's look at Sample 4 again.
    # Sample 4: 2 2 2 1 3 3 3 1 1.
    # This is: mid repeated K times, then 1 repeated K times, then N repeated K times...
    # No, it's: mid(K), 1(K), N(K), 2(K), N-1(K)...
    # Let's check Sample 4: 2(3), 1(3), 3(3) -> 2 2 2 1 1 1 3 3 3.
    # Still not 2 2 2 1 3 3 3 1 1.
    
    # Let's use the property: the median sequence A satisfies A_i + A_{NK-i+1} = N+1.
    # To be the floor((S+1)/2)-th, A must be the lexicographically smallest sequence 
    # such that A >= A_complement.
    # This means at the first index i where A_i != A_complement_i, we must have A_i < A_complement_i
    # is FALSE. So A_i >= A_complement_i.
    # To make A smallest, we want A_i to be as small as possible.
    # For the first index i, we want A_i to be the smallest value such that 
    # it's possible to complete the sequence with A >= A_complement.
    # That value is A_1 = (N + 1) // 2.
    # Then for i = 2 to NK/2, we want A_i to be as small as possible.
    # The smallest possible A_i is 1, then 2, etc.
    # But we must ensure A_i <= A_{NK-i+1} (since A_i + A_{NK-i+1} = N+1).
    # So for i = 2, 3, ..., NK/2:
    # A_i is the smallest available number.
    # A_{NK-i+1} is then N + 1 - A_i.
    # This uses up the counts of the numbers.
    
    # Implementation using reduce to avoid loops:
    def get_sequence(N, K):
        # state: (current_counts, result_list)
        # We only need to determine the first half of the sequence.
        # The second half is mirrored.
        
        # Initial counts of each number 1...N
        initial_counts = {i: K for i in range(1, N + 1)}
        
        # First element
        mid = (N + 1) // 2
        res = [0] * (N * K)
        res[0] = mid
        counts_after_first = {i: (K - 1 if i == mid else K) for i in range(1, N + 1)}
        
        # We need