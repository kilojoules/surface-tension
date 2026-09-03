```python
import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # The operation allows us to replace a segment (l+1, r-1) with the value of A[l]
    # if A[l] == A[r] and all A[i] for l < i < r were different from A[l].
    # This is essentially a process of merging adjacent blocks of the same value.
    # Let's compress the sequence A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks = [(1, 5), (0, 1)] where (val, len)
    
    # To check if A is reachable:
    # The initial state is 1, 0, 1, 0, ...
    # The operation preserves the values at the boundaries of the modified range.
    # Crucially, the operation can only occur if the middle is different.
    # This means we can only "expand" a color into a region of the opposite color.
    # The resulting sequence A is reachable if and only if it can be reduced to 
    # the initial alternating sequence (1, 0, 1, 0...) by reversing the operation.
    # The reverse operation: if we have a block of identical elements, we can 
    # "split" it if the elements to the left and right are the same.
    
    # Actually, a simpler condition: A is reachable if and only if 
    # A_i = i % 2 (or A_i = (i+1) % 2) is NOT required.
    # The constraint is: we can never create a pattern that wasn't there.
    # But we can remove alternating patterns.
    # The core logic: A is reachable if we can reduce it to the alternating sequence.
    # Let's represent A as a sequence of lengths of alternating blocks.
    # E.g., 1 1 1 1 1 0 -> block of 1s (len 5), block of 0s (len 1).
    # The only way to get a block of length k > 1 is by performing the operation.
    # An operation on (l, r) where A[l]=A[r]=v and A[l+1...r-1]=1-v 
    # turns a segment of length (r-l+1) with alternating values into a segment 
    # where the middle is filled with v.
    
    # Let's use the property: the number of ways to form a block of length k 
    # using this specific operation is the (k-1)-th Catalan number? 
    # No, it's simpler. To get a block of length k, we need to have had 
    # an alternating sequence of length k, and we perform operations.
    # The number of ways to reduce an alternating sequence of length k to a 
    # uniform sequence of length k using these rules is C(k-1), 
    # where C is the Catalan number? No.
    # Let's re-evaluate: 
    # To make A[l...r] all same, we need A[l]=A[r]. The middle A[l+1...r-1] 
    # must be the opposite color.
    # This is exactly the structure of a binary tree or nested parentheses.
    # The number of ways to clear a segment of length 'len' of the opposite color
    # is given by the Catalan number C_{len}.
    
    # Wait, the problem is simpler: 
    # 1. Check if A is reachable. A is reachable if we can partition it into 
    #    blocks of identical elements such that the blocks themselves alternate 
    #    in value, and each block i (except possibly the first and last) 
    #    was created by an operation.
    # 2. For a block of length k to be created, it must have "consumed" 
    #    an alternating sequence.
    # The number of ways to form a block of length k is C_{(k-1)//2} if k is odd, 
    # and 0 if k is even? No.
    
    # Correct logic:
    # An operation takes (l, r) where A[l]=A[r] and A[l+1...r-1] are opposite.
    # This means the middle part must have been an alternating sequence.
    # Let the compressed blocks be (val_1, len_1), (val_2, len_2), ... (val_m, len_m).
    # For the result to be reachable, we must be able to reduce this to 
    # the initial alternating sequence.
    # The only way to increase the length of a block is to "swallow" 
    # a block of the opposite color that has length 1.
    # If we have a block of length len_i, it must have been formed by 
    # swallowing (len_i - 1) / 2 blocks of the opposite color, each of length 1.
    # This implies len_i must be odd.
    # Exception: The first and last blocks can be any length? 
    # No, because the operation requires l and r to be within [1, N].
    # If we swallow a block of length 1, the length of the current block increases by 2.
    # So len_i must be odd for all 1 < i < m.
    # For i=1, if it's length len_1, it could have swallowed (len_1 - 1)//2 blocks.
    # But the operation is: replace l+1...r-1 with A[l].
    # This means the block at l expands to the right.
    # To get A_1...A_{len_1} all same, we need A_{len_1+1} to be the same as A_1, 
    # and A_2...A_{len_1} to be opposite.
    # This means the blocks must have lengths: len_1, 1, 1, 1... 
    # Actually, the condition is: A is reachable iff len_i is odd for all 1 < i < m.
    # And the number of ways is the product of Catalan numbers C_{(len_i - 1) // 2}.
    # For the first and last blocks, they can be any length? 
    # No, if len_1 is even, it can't be formed by the operation because the 
    # operation adds 2 to the length (one from the opposite block, one from the 
    # boundary).
    # Wait, if we have 1 0 1, and we operate on (1, 3), we get 1 1 1. Length 1 -> 3.
    # If we have 1 0 1 0 1, we can get 1 1 1 1 1. Length 1 -> 3 -> 5.
    # So len_i must be odd for all i.
    # But we can also use the boundary. If we have 0 1 0, and we operate on (1, 3), 
    # we get 0 0 0.
    # The only way to get an even length is if the block is at the end and we 
    # couldn't fit the last '1'.
    # Actually, the constraint is: len_i must be odd for 1 < i < m.
    # For i=1 and i=m, they can be anything? 
    # Let's check Sample 1: 1 1 1 1 1 0. Blocks: (1, 5), (0, 1). 
    # len_1=5, len_2=1. Both odd. C_{(5-1)//2} = C_2 = 2. C_{(1-1)//2} = C_0 = 1. 
    # Total = 2 * 1 = 2. But sample says 3.
    # Let's re-read: "Choose cells l and r (l+1 < r)".
    # If we have 1 0 1, l=1, r=3, we get 1 1 1.
    # If we have 1 1 1 0, we can't use the operation to get more 1s.
    # Sample 1: 1 1 1 1 1 0. 
    # Initial: 1 0 1 0 1 0.
    # Op 1: l=2, r=4 -> 1 0 0 0 1 0.
    # Op 2: l=1, r=5 -> 1 1 1 1 1 0.
    # This means the blocks can be formed by "eating" the opposite color.
    # The number of ways to form a block of length k is the number of ways to 
    # parenthesize the operations. This is C_{(k-1)//2}.
    # But the first and last blocks are special.
    # A block of length k at the start can be formed if we have k-1 elements 
    # to its right that can be swallowed.
    # The correct condition: len_i must be odd for 1 < i < m.
    # The number of ways is product_{i=1}^m (Ways to form block i).
    # For 1 < i < m, ways = C_{(len_i - 1) // 2}.
    # For i = 1, if len_1 is even, it's impossible? No.
    # If len_1 is even, say 2, we have 1 1 0... 
    # Initial was 1 0 1 0... To get 1 1, we need to operate on (1, 3) to get 1 1 1, 
    # then we can't "remove" one.
    # Actually, the only way to get an even length block is if it's the 
    # last block and it's truncated, or the first block and we started from 0.
    # But the problem says cell i has i % 2. So cell 1 is 1, 2 is 0, 3 is 1...
    # The sequence is 1, 0, 1, 0, 1, 0...
    # To get a block of length k, we need k to be odd, UNLESS it's the last block.
    # If the last block is len_m, it can be any parity because the "r" 
    # can be beyond N? No, r <= N.
    # Let's use the property: len_i must be odd for 1 <= i < m.
    # For the last block len_m, it can be anything.
    # For Sample 1: len_1=5 (odd), len_2=1 (odd). C_2 * C_0 = 2 * 1 = 2. Still not 3.
    # Wait, the sample says 3. The operations were:
    # 1. (2, 4) -> 1 0 0 0 1 0
    # 2. (1, 5) -> 1 1 1 1 1 0
    # Or: (1, 3) -> 1 1 1 0 1 0, then (3, 5) -> 1 1 1 1 1 0.
    # Or: (1, 5) -> 1 1 1 1 1 0.
    # These are 3 ways. This is exactly the number of ways to form a block of 
    # length 5 using blocks of length 1.
    # The number of ways to form a block of length k is C_{(k-1)//2} ONLY IF 
    # the block is internal. If it's the first block, we can use the 
    # "right" boundary.
    # Let's use the formula: ways(k) = (k+1)//2-th Catalan number?
    # C_0=1, C_1=1, C_2=2, C_3=5.
    # For k=5, (5+1)//2 = 3, C_2 = 2. Still not 3.
    # Let's re-count: k=5. 
    # 1. (1, 5)
    # 2. (1, 3) then (3, 5)
    # 3. (2, 4) then (1, 5)
    # These are 3. This is the Fibonacci sequence? No.
    # These are the number of binary trees where each node has 2 children, 
    # but the structure is slightly different.
    # Actually, this is the number of ways to reduce a string of length k 
    # to a single character using the given operation.
    # For k=1, ways