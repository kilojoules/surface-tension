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

    # The operation is: choose l, r such that X_l == X_r and X_i != X_l for l < i < r.
    # Then X_i becomes X_l for l < i < r.
    # This is equivalent to saying we can merge a block of alternating colors 
    # into a single color if the boundaries match.
    # Specifically, this problem can be modeled as counting the number of ways 
    # to reduce the initial string (101010...) to the target string A 
    # using these "fill" operations.
    
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # The initial state is X = [1, 0, 1, 0, 1, 0, ...]
    # Note: The problem says cell i has i % 2. 
    # So X_1 = 1, X_2 = 0, X_3 = 1, ...
    
    # If A is not reachable from X, the answer is 0.
    # A is reachable if and only if A can be formed by replacing 
    # contiguous segments of X with the value of the segment's endpoints.
    # This means A must be a "contraction" of X.
    # However, a simpler observation: 
    # The operation removes "peaks" or "valleys" in the sequence.
    # The number of ways to reach A is related to the number of ways to 
    # parenthesize the reductions.
    
    # Let's look at the blocks of A. 
    # Let the blocks be (val_1, len_1), (val_2, len_2), ... (val_k, len_k).
    # For each block j, it must have been derived from a segment of X.
    # The only way to get a block of length L of color C is to start with 
    # a segment of X that starts and ends with C and has length L, 
    # or to expand via operations.
    
    # Actually, the core of this problem is:
    # Each block of identical characters in A (except possibly the first and last)
    # must have been created by operations.
    # If we have a block of length L, it takes (L-1) "merges" to create it 
    # if we consider the underlying X structure.
    # The number of ways to form a block of length L using these operations 
    # is the Catalan number C_{L-1} if we view it as a binary tree of operations.
    # But the operations here are specific.
    
    # Let's re-evaluate:
    # To get a block of length L of color '1', we need a sequence 1, 0, 1, 0, 1...
    # The number of ways to reduce a sequence of length 2k-1 (1,0,1,0,1) to (1,1,1,1,1)
    # is the Catalan number C_{k-1}.
    # The total number of ways is the product of C_{(len_i + 1)//2 - 1} for each block,
    # provided the blocks are consistent with the alternating pattern of X.
    
    # Check consistency:
    # X_i = i % 2.
    # A_i must be reachable. This means we cannot have A_i = A_{i+1} 
    # unless that transition was possible.
    # Actually, the only constraint is that we can't "create" a color 
    # that wasn't there. But X has both.
    # The real constraint: to turn X[l...r] into all X[l], 
    # we need X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This means the segment X[l...r] must be (0, 1, 0) or (1, 0, 1).
    # This operation reduces the length of the sequence of alternating colors.
    
    # Let's use the property: 
    # A block of length L in A corresponds to a segment of length 2L-1 in X 
    # (if we reduce it fully).
    # Wait, the sample 1: N=6, A=[1,1,1,1,1,0]. 
    # X = [1,0,1,0,1,0]. 
    # A consists of a block of 1s (len 5) and a block of 0s (len 1).
    # The block of 1s comes from X[1...5] = [1,0,1,0,1].
    # The number of ways to reduce [1,0,1,0,1] to [1,1,1,1,1] is C_2 = 2.
    # No, sample says 3. Let's see:
    # Ops: (2,4) then (1,5) OR (4,6) is not possible since A_6=0.
    # (2,4) -> [1,0,0,0,1,0], then (1,5) -> [1,1,1,1,1,0]
    # (4,6) is not allowed because A_6=0.
    # Wait, (2,4) then (1,5) is one.
    # Also (3,5) then (1,5) is two.
    # Also (2,4) and (3,5) are the two ways to get [1,0,0,0,1,0] or [1,0,1,1,1,0].
    # Then (1,5) finishes it.
    # Actually, the number of ways to reduce a segment of length 2k-1 to 1s is C_{k-1}.
    # For L=5, k=3, C_2 = 2. 
    # But we also have the case where we don't reduce fully? No, A is fixed.
    # Let's use the formula: The number of ways is the product of 
    # (2k-2)! / (k! (k-1)!) where k is the number of elements of the opposite color 
    # removed to form the block.
    
    # Correct logic:
    # A block of length L of color C is formed by taking a segment of X 
    # that starts and ends with C and contains (L-1) elements of the opposite color.
    # The number of ways to clear these (L-1) elements is the Catalan number C_{L-1}.
    # However, the blocks in A must be contiguous.
    # Let the blocks of A be B_1, B_2, ..., B_m with lengths L_1, L_2, ..., L_m.
    # The total number of ways is Product(C_{L_i - 1})? 
    # No, that's for a different problem.
    
    # Let's use the property: each operation reduces the number of blocks of 
    # identical consecutive elements by 2.
    # Initial X has N blocks. Final A has m blocks.
    # Number of operations = (N - m) / 2.
    # This is only possible if N and m have the same parity.
    
    # The number of ways is actually simpler:
    # For each block i of length L_i, the number of ways to form it is 
    # the number of binary trees with L_i leaves, which is C_{L_i - 1}.
    # But we must ensure the blocks in A are consistent with X.
    # X = 1, 0, 1, 0, ...
    # A_i must be X_i if A_i is the start of a block that isn't "filled".
    # Actually, the only requirement is that A must be reachable.
    # A is reachable if we can partition A into segments S_1, ..., S_m 
    # such that each S_j consists of identical elements and the 
    # concatenated sequence of the first elements of each S_j is exactly X_1, X_2, ...
    # No, that's not it.
    
    # Let's use the known result for this specific problem:
    # The answer is the product of C_{(L_i - 1)} where L_i are the lengths of 
    # blocks of identical elements in A, BUT only for blocks that 
    # "covered" some elements of the opposite color.
    # If A = [1, 1, 1, 1, 1, 0], blocks are [1]*5 and [0]*1.
    # L_1 = 5, L_2 = 1.
    # C_{5-1} = C_4 = 14? No.
    # The number of elements of the opposite color inside the block of length L_i 
    # is (L_i - 1) // 2 if the block is aligned with X.
    
    # Let's re-read: X_i = i % 2. (X_1=1, X_2=0, X_3=1...)
    # A block of length L of color C starting at index i.
    # It covers X_i, X_{i+1}, ..., X_{i+L-1}.
    # For this to be possible, X_i must be C and X_{i+L-1} must be C.
    # This means L must be odd.
    # If L is odd, the number of ways to form it is C_{(L-1)//2}.
    # If any L_i is even, it's impossible? No, because a block can be 
    # extended by absorbing an adjacent block.
    
    # Correct approach:
    # A is reachable if and only if A_i = X_i for all i such that 
    # A_i != A_{i-1} (the start of each block must match X).
    # Then the answer is Product(C_{(L_i - 1)//2}) where L_i are lengths of blocks.
    # But L_i must be odd. If any L_i is even, the answer is 0.
    # Wait, Sample 1: A = [1, 1, 1, 1, 1, 0]. Blocks: L_1=5 (color 1), L_2=1 (color 0).
    # L_1=5 is odd, L_2=1 is odd. C_{(5-1)//2} * C_{(1-1)//2} = C_2 * C_0 = 2 * 1 = 2.
    # Still not 3. Let me re-read.
    # Sample 1: X = 1 0 1 0 1 0. A = 1 1 1 1 1 0.
    # We can do (2,4) then (1,5) -> 3 ways.
    # The operations are:
    # 1. (2,4) then (1,5)
    # 2. (3,5) then (1,5)
    # 3. (2,4) and (3,5) are not disjoint.
    # Actually, the two ways to get [1,1,1,1,1] from [1,0,1,0,1] are:
    # Op(2,4) -> [1,0,0,0,1], then Op(1,5) -> [1,1,1,1,1]
    # Op(3,5) -> [1,0,1,1,1], then Op(1,5) -> [1,1,1,1,1]
    # Op(1,5) is only possible if X_1 == X_5 and X_2,3,4 != X_1.
    # But after Op(2,4), X becomes [1,0,0,0,1], so X_2,3,4 are all 0, which is != X_1.
    # So Op(1,5) is possible.
    # Total ways = 2 + 1 = 3. This is C_3? No.
    # The number of ways to reduce a sequence of length 2k-1 to a single color is 2^{k-2} ? No.
    # For k=3 (length 5), it's 3. For k=2 (length 3), it's 1. For k=1 (length 1), it's 1.
    # The sequence 1, 1, 3, 11, 4