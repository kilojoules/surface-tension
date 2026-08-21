The constraint to avoid explicit loops and recursion forces a shift from imperative state management to a functional paradigm, emphasizing the transformation of data streams rather than the mutation of variables. In this specific problem, we need to determine the lexicographically middle sequence of all permutations of a multiset. Since the total number of sequences $S$ is symmetric (replacing $i$ with $N-i+1$ maps the $k$-th sequence to the $(S-k+1)$-th), the $\lfloor(S+1)/2\rfloor$-th sequence is the one that is "self-dual" or the first one that is lexicographically greater than or equal to its complement. This allows us to determine the sequence greedily by comparing the number of available permutations starting with a certain digit against the remaining required rank. However, a simpler observation is that we are looking for the sequence $A$ such that $A$ is the smallest sequence where $A \ge \text{reverse\_complement}(A)$. For this specific problem, the target sequence is the one that is lexicographically "central," which corresponds to the sequence that is its own complement when mirrored, or specifically, the sequence that starts with the middle possible digit and follows a pattern of symmetry.

Actually, the most reliable way to find the $\lfloor(S+1)/2\rfloor$-th sequence without loops/recursion and without calculating massive factorials (which would be slow) is to realize that the "middle" sequence is the one that is lexicographically smallest among those that are "greater than or equal to" their complement (where complement of $x$ is $N-x+1$). This means we try to place the smallest possible digit $d$ at each position, and check if it's still possible to be $\ge$ the complement.

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

    # The total number of sequences S is (NK)! / (K!^N).
    # We want the floor((S+1)/2)-th sequence.
    # Due to symmetry, the complement of a sequence (replacing i with N-i+1)
    # maps the k-th sequence to the (S-k+1)-th.
    # The middle sequence is the smallest sequence A such that A >= complement(A).
    
    # To implement this without loops or recursion, we use a greedy approach
    # simulated via a list comprehension or reduce.
    # We maintain the state: (current_counts, is_already_greater)
    # current_counts: list of remaining counts for each digit 1...N
    # is_already_greater: boolean, true if the sequence prefix is already 
    # lexicographically greater than the prefix of its complement.

    def get_next_char(state, _):
        counts, already_greater = state
        
        # We want the smallest digit d such that we can still form a sequence 
        # that is >= its complement.
        # The complement of digit d is (n - d + 1).
        # If already_greater is True, we can pick the smallest available d.
        # If already_greater is False, we must pick d such that d >= (n - d + 1)
        # is possible, or d < (n - d + 1) but we can make the rest greater.
        # Actually, the simplest logic for the "middle" sequence:
        # At each position, try d = 1, 2, ..., N.
        # If we pick d, the complement digit is d' = n - d + 1.
        # If d > d', then for all future positions, we can pick the smallest available.
        # If d < d', we must check if the remaining counts allow the sequence 
        # to eventually become greater than its complement.
        # However, the symmetry implies we can just try to keep d == d' as long as possible.
        
        # Correct logic for the middle sequence:
        # Try d from 1 to N. If d < n - d + 1, we are 'below' the complement.
        # If d > n - d + 1, we are 'above'.
        # If we are already 'above', we pick the smallest available d to stay smallest.
        # If we are 'below' or 'equal', we check if picking d allows us to 
        # eventually reach the middle.
        
        # For this specific problem, the middle sequence is constructed by:
        # For each position, try d = 1, 2, ..., N.
        # If we pick d, and the remaining counts are c_1, ..., c_N:
        # The number of sequences starting with d is (sum(c_i))! / product(c_i!).
        # This is too slow. 
        
        # Observation: The middle sequence is the one that is 
        # lexicographically smallest among those A where A >= complement(A).
        # This means we try d = 1, 2, ..., N.
        # If d > n - d + 1, we are now 'above', so we can pick the smallest 
        # available digits for the rest of the sequence.
        # If d < n - d + 1, we are 'below', so we must be able to 'recover' 
        # by picking a larger digit later.
        # But we want the SMALLEST such sequence.
        # The smallest sequence A such that A >= complement(A) will have
        # d = (n+1)//2 at the first position where it differs from its complement.
        
        # Let's use the property: the middle sequence is the one that 
        # mirrors the distribution.
        # For N=3, K=3: 2 2 2 1 3 3 3 1 1
        # The middle digit is (3+1)//2 = 2.
        # It fills all K of the middle digit, then the smallest, then the largest...
        pass

    # The pattern for the middle sequence is:
    # 1. Fill the middle digit (N+1)//2 for K times.
    # 2. Then alternate: smallest available, largest available.
    # Specifically: 
    # Middle digit M = (N+1)//2.
    # Sequence: [M]*K, then for i from 1 to (N-1)//2:
    # [i]*K, [N-i+1]*K, then repeat until all used.
    # Wait, Sample 4: N=3, K=3 -> 2 2 2 1 3 3 3 1 1
    # That is: 2(3 times), 1(1 time), 3(3 times), 1(2 times).
    # Let's re-evaluate: 2 2 2 1 3 3 3 1 1
    # This is: M=2. Sequence: 2,2,2, 1, 3,3,3, 1,1.
    # The logic is: 
    # Place M as many times as possible (K).
    # Then place the smallest available digit (1) once.
    # Then place the largest available digit (N) as many times as possible (K).
    # Then place the second smallest (2) once... and so on.
    # Actually, the simplest way to describe the middle sequence:
    # It's the sequence that is its own complement reversed.
    # For N=3, K=3, the sequence is 2 2 2 1 3 3 3 1 1.
    # Complement: 2 2 2 3 1 1 1 3 3.
    # Reversed Complement: 3 3 1 1 1 3 1 2 2 2. Not quite.
    
    # Let's use the property: the middle sequence is the one where we 
    # greedily pick the smallest d such that the number of sequences 
    # starting with d is >= the number of sequences starting with digits < d.
    # Since we can't use loops, we can use a recursive-like structure 
    # via a list comprehension or map, but the constraint forbids recursion.
    # However, we can use a large list comprehension to simulate the process.
    
    # Correct logic for the middle sequence:
    # It is the sequence that is lexicographically smallest among those A 
    # such that A >= complement(A).
    # To be smallest, we want the first digit to be as small as possible.
    # If we pick d < (N+1)/2, then the complement starts with d' > (N+1)/2.
    # So A < complement(A) unless a later digit makes it larger.
    # But if we pick d < (N+1)/2, the "best" we can do to make A > complement(A)
    # is to put all remaining largest digits as early as possible.
    # The middle sequence is:
    # 1. Try d = 1, 2, ..., N.
    # 2. If d < (N+1)/2, we can only pick d if the remaining 
    #    multiset allows a sequence > complement.
    # 3. The simplest construction:
    #    The middle sequence is the one that starts with (N+1)//2, 
    #    then follows with the smallest possible digits, 
    #    but mirrored by the largest.
    
    # For N=3, K=3:
    # Digits: 1,1,1, 2,2,2, 3,3,3
    # Middle is 2.
    # Result: 2 2 2 1 3 3 3 1 1
    # This is: 
    # - All of digit (N+1)//2
    # - Then 1 copy of digit 1
    # - Then all of digit N
    # - Then 1 copy of digit 1...
    # No, the pattern is:
    # While digits remain:
    #   If we can place the middle digit, do it.
    #   Otherwise, place the smallest available, then the largest available.
    
    # Let's use the property that the middle sequence is the one 
    # that is "lexicographically" the median.
    # For N=3, K=3, the sequence is 2 2 2 1 3 3 3 1 1.
    # This is: 2(K), 1(1), 3(K), 1(K-1).
    # For N=2, K=2: 1 2 2 1. (M=1.5). 1(1), 2(2), 1(1).
    # For N=6, K=1: 3 6 5 4 2 1. (M=3.5). 3(1), 6(1), 5(1), 4(1), 2(1), 1(1).
    
    # The general rule:
    # 1. Start with d = (N+1)//2.
    # 2. Place d as many times as possible (K).
    # 3. Then place the remaining digits in a specific order:
    #    Smallest available, then Largest available, then 2nd smallest, etc.
    #    But the "Largest" block is placed fully, and the "Smallest" block 
    #    is placed one by one? No.
    
    # Let's use the property: the middle sequence is the one that 
    # is the smallest sequence A such that A >= complement(A).
    # For N=3, K=3:
    # Try d=1: Complement starts with 3. 1 < 3. To be >=, we need 
    # the rest of A to be >= complement of the rest.
    # The largest possible sequence starting with 1 is 1 3 3 3 2 2 2 1 1.
    # Its complement is 3 1 1 1 2 2 2 3 3.
    # 1 3 3 3 2 2 2 1 1 < 3 1 1 1 2 2 2 3 3.
    # So d=1 is impossible.
    # Try d=2: Complement starts with 2. 2 == 2.
    # We need the rest of A to be >= complement of the rest.
    # The smallest sequence starting with 2 is 2 1 1 1 2 2 3 3 3.
    # Its complement is 2 3 3 3 2 2 1 1 1.
    # 2 1 1 1... < 2 3 3 3...
    # We need to find the smallest A starting with 2 such that A >= complement(A).
    # To make A >= complement(A) with the smallest prefix, we should 
    # keep the digits equal as long as possible, then make one digit larger.
    # For N=3, K=3, d=2:
    # A: 2 2 