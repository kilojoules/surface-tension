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
    # This is exactly the "middle" sequence.
    # Because the set of good sequences is symmetric (replacing x with N-x+1),
    # the middle sequence is the one that is "halfway" through the lexicographical order.
    # A key property of lexicographical symmetry is that the sequence at index (S+1)//2
    # is the one that, when concatenated with its "complement" (reversed and mirrored),
    # forms a symmetric pair.
    # More simply: the middle sequence is the one that is lexicographically 
    # just before the point where the first element switches from 1 to 2, 
    # or 2 to 3, etc., across the total distribution.
    
    # For a balanced distribution, the middle sequence is the one that 
    # effectively "balances" the counts.
    # The sequence is: 
    # For i from 1 to N:
    #   If i < (N+1)/2: output i, K times.
    #   If i == (N+1)/2 (and N is odd): output i, K times.
    #   If i > (N+1)/2: output i, K times.
    # Wait, that's just 1...N. That's the 1st sequence.
    
    # Let's reconsider: The total number of sequences is S.
    # The sequence at index (S+1)//2 is the one that is "central".
    # If we map each sequence Seq to its complement Seq' where Seq'_i = N + 1 - Seq_i,
    # then Seq < Seq' if and only if the first index i where they differ has Seq_i < Seq'_i.
    # The middle sequence is the one where Seq is "closest" to its complement.
    # Specifically, the sequence that is the lexicographical median.
    
    # For N=2, K=2: S=6. (S+1)//2 = 3. Sequences:
    # 1: 1,1,2,2 | 6: 2,2,1,1
    # 2: 1,2,1,2 | 5: 2,1,2,1
    # 3: 1,2,2,1 | 4: 2,1,1,2
    # Result: 1,2,2,1.
    
    # For N=6, K=1: S=720. (S+1)//2 = 360.
    # This is the last sequence starting with 1, 2, and 3.
    # Specifically, the last sequence that is lexicographically smaller than 
    # its complement.
    # The complement of (S1, ..., Sn) is (N+1-S1, ..., N+1-Sn).
    # The middle sequence is the largest sequence Seq such that Seq < Complement(Seq).
    # To make Seq as large as possible while Seq < Complement(Seq):
    # The first element S1 must be <= (N+1)/2.
    # To maximize Seq, we want S1 to be the largest possible value such that 
    # there exists a sequence starting with S1 that is smaller than its complement.
    # If N is even, S1 = N//2. If N is odd, S1 = (N+1)//2.
    # However, if S1 = (N+1)//2, then for Seq < Complement(Seq), 
    # the remaining sequence must be smaller than its complement.
    # To maximize the sequence, we fill the remaining slots with the largest 
    # available numbers in descending order, as long as the overall 
    # sequence remains smaller than its complement.
    
    # Correct logic for the median of all permutations of a multiset:
    # The sequence is the largest sequence Seq such that Seq < Complement(Seq).
    # 1. Determine the first index i where Seq_i != Complement(Seq)_i.
    # 2. For the median, we want the first element S_1 to be as large as possible
    #    such that S_1 < N + 1 - S_1. 
    #    This means S_1 <= N // 2.
    #    So S_1 = N // 2.
    # 3. Once S_1 is fixed at N // 2, to make the sequence as large as possible,
    #    we want the remaining elements to be as large as possible.
    #    The remaining elements are: 
    #    - (N // 2) appears (K-1) more times.
    #    - 1 to (N // 2 - 1) appear K times.
    #    - (N // 2 + 1) to N appear K times.
    #    To maximize the sequence, we place the largest available numbers first.
    #    BUT, we must ensure the total sequence is still < its complement.
    #    Since S_1 (N // 2) < Complement(S_1) (N + 1 - N // 2), 
    #    the condition Seq < Complement(Seq) is already satisfied!
    #    Therefore, we can simply place all remaining elements in descending order.
    
    # Example N=2, K=2: S_1 = 2 // 2 = 1. Remaining: {1:1, 2:2}.
    # Descending: 2, 2, 1. Result: 1, 2, 2, 1. (Correct)
    # Example N=6, K=1: S_1 = 6 // 2 = 3. Remaining: {1:1, 2:1, 3:0, 4:1, 5:1, 6:1}.
    # Descending: 6, 5, 4, 2, 1. Result: 3, 6, 5, 4, 2, 1. (Correct)
    # Example N=3, K=3: S_1 = 3 // 2 = 1. 
    # Wait, if N=3, N//2 = 1. But the sample says 2 2 2...
    # Let's re-evaluate. If N=3, the elements are 1, 2, 3.
    # The complement of 2 is 3+1-2 = 2.
    # If S_1 = 2, then S_1 == Complement(S_1). 
    # Then we look at S_2. If S_2 < Complement(S_2), the sequence is smaller.
    # To maximize the sequence, we can have S_1 = 2, S_2 = 2, S_3 = 2.
    # Then we look at S_4. The remaining elements are {1:3, 3:3}.
    # To keep Seq < Complement(Seq), we need the first differing element to be smaller.
    # The first differing element will be S_4. We want S_4 to be as large as possible
    # but still S_4 < Complement(S_4).
    # The available elements for S_4 are 1 and 3. The complement of 1 is 3.
    # So S_4 must be 1.
    # Then the remaining elements {1:2, 3:3} can be placed in descending order: 3, 3, 3, 1, 1.
    # Result: 2, 2, 2, 1, 3, 3, 3, 1, 1. (Correct)

    # General Algorithm:
    # 1. Start with counts = {i: K for i in range(1, N + 1)}
    # 2. We want the largest sequence Seq such that Seq < Complement(Seq).
    # 3. For each position, try the largest possible value v from N down to 1.
    # 4. If v < (N + 1) - v, then this v makes Seq < Complement(Seq).
    #    Since we want the largest such sequence, we can pick this v and then
    #    fill all remaining positions with the largest available numbers in descending order.
    # 5. If v == (N + 1) - v, we can pick v and continue to the next position to 
    #    determine if the sequence will be < or > its complement.
    # 6. If v > (N + 1) - v, we cannot pick this v as the first differing element.
    #    But we can only pick v if we have already established that a previous 
    #    element S_i was < Complement(S_i). 
    #    However, we are looking for the FIRST index where they differ.
    #    So if all previous S_i == Complement(S_i), we cannot pick v > (N+1)-v.

    # Refined Algorithm:
    # While elements remain:
    #   Find the largest v in [1, N] such that count[v] > 0 and:
    #   a) v < (N + 1) - v  => This is the first difference. We can take this v,
    #      then fill the rest descending.
    #   b) v == (N + 1) - v => This is not a difference. We can take this v and
    #      move to the next position.
    #   c) v > (N + 1) - v  => This would make Seq > Complement(Seq). 
    #      We cannot take this as the first difference.
    
    # Since we want the largest sequence, we first try to take as many v == (N+1)-v 
    # as possible, then one v < (N+1)-v, then the rest descending.
    
    mid = (N + 1) / 2
    # Count of the middle element (only exists if N is odd)
    # If N is even, there is no v == N + 1 - v.
    # If N is odd, v = (N+1)//2 is the middle element.
    
    res = []
    counts = {i: K for i in range(1, N + 1)}
    
    # 1. Take all possible middle elements
    if N % 2 != 0:
        m_val = (N + 1) // 2
        for _ in range(K):
            res.append(m_val)
        counts[m_val] = 0
        
    # 2. Take the largest v < mid
    # The values < mid are 1, ..., (N-1)//2
    # The largest is (N-1)//2 if N > 0.
    # But we must handle N=1 separately.
    if N > 0:
        # We need to find the largest v < mid that has count > 0.
        # Since we haven't used any v < mid yet, it's just floor((N-1)/2) if N > 1.
        # Wait, if N=1, the loop for middle elements already took everything.
        # If N > 1:
        v_small = (N // 2) if (N % 2 == 0) else (N // 2)
        # Actually, the largest v such that v < N + 1 - v is always N // 2.
        # Let's use that.
        
        # Correct logic:
        # If N is even, the "middle" elements are N//2 and N//2 + 1.
        # The first difference must be S_i < Complement(S_i).
        # To maximize, we want S_i to be as large as possible, so S_i = N // 2.
        # Before that, we can have any S_j == Complement(S_j).
        # But S_j == Complement(S_j) only if S_j = (N+1)/2.
        # This only happens if N is odd.
        
        # So:
        # 1. If N is odd, output (N+1)//2, K times.
        # 2. Output N // 2, 1 time.
        # 3. Output all remaining elements in descending order.
        
        # Let's trace N=2, K=2:
        # N is even. Step 1: nothing. Step 2: 2//2 = 1. Step 3: remaining {1:1, 2:2} desc -> 2, 2, 1.
        # Result: 1, 2, 2, 1. (Correct)
        
        # Trace N=6, K=1:
        # N is even. Step 1: nothing. Step 2: 6//2 = 3. Step 3: remaining