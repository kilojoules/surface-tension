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

    # The operation allows us to take two cells l and r with the same value
    # and overwrite everything in between with that value, provided the 
    # middle elements were different.
    # This is essentially a process of merging contiguous blocks of the same value.
    # Let's represent the grid as a sequence of blocks (value, length).
    # Initial state: (1, 1), (0, 1), (1, 1), (0, 1) ...
    # Target state: A
    
    # First, check if the target state A is reachable.
    # The operation preserves the values at the boundaries of the range [l, r].
    # Specifically, A[0] must be 1 % 2 = 1 and A[N-1] must be N % 2.
    # Wait, the problem says cell i is i mod 2. 
    # Let's use 0-indexed: cell i (1-indexed) is i % 2.
    # A[0] (1st cell) = 1 % 2 = 1.
    # A[N-1] (N-th cell) = N % 2.
    
    # If A[0] != 1 or A[N-1] != (N % 2), it's impossible.
    if A[0] != 1 or A[N-1] != (N % 2):
        print(0)
        return

    # We can group A into contiguous blocks of identical values.
    # Example: 1 1 1 1 1 0 -> (1, 5), (0, 1)
    # The number of blocks in the initial state is N.
    # Each operation reduces the number of blocks.
    # Specifically, if we have blocks ... B_{k-1}, B_k, B_{k+1} ...
    # and B_{k-1} and B_{k+1} have the same value, and B_k has a different value,
    # performing the operation on the boundaries of B_k merges B_{k-1}, B_k, and B_{k+1} into one.
    # This reduces the block count by 2.
    
    # Let the target sequence A be compressed into M blocks.
    # The initial sequence is 1, 0, 1, 0 ... (N blocks).
    # To reach M blocks from N blocks, we need (N - M) / 2 operations.
    # If (N - M) is odd, it's impossible.
    
    # To count the sequences:
    # This is equivalent to counting ways to parenthesize the reduction of blocks.
    # Each operation removes one block of value X and merges it into surrounding blocks of value Y.
    # This is like a binary tree structure or Catalan-like counting.
    # For a sequence of M blocks, the number of ways to form it is the product of
    # Catalan numbers C_k where k is the number of blocks removed to form each final block.
    # Actually, a simpler way:
    # Each final block i (except the last) is followed by a block of different value.
    # The number of ways to form the final configuration is the product of 
    # Catalan( (len_initial_segment - 1) // 2 ) for each segment.
    # But the blocks are interleaved.
    
    # Let's use the property: to get a block of length L of value V, 
    # we must have started with a sequence of length L of alternating values.
    # The number of ways to reduce a sequence of length 2k+1 (V, !V, V, !V, ..., V) 
    # to a single block of value V is the k-th Catalan number C_k.
    
    # The target A consists of blocks of lengths L_1, L_2, ..., L_M.
    # The first block is 1s, second is 0s, etc.
    # For the i-th block to be formed, it must have been an alternating sequence 
    # of the same length L_i, and we reduce it to one value.
    # This is only possible if L_i is odd. If any L_i is even, 0 ways.
    # Exception: The last block can be merged into the previous one, but the 
    # operation requires l+1 < r, so the middle is at least one element.
    # Actually, the only way to change the number of blocks is to remove a block 
    # of length 1 (or more) and merge two blocks of the same value.
    
    # Correct logic:
    # The target A is reachable if and only if it can be represented as 
    # A = (1 * L_1, 0 * L_2, 1 * L_3, ..., V * L_M) where each L_i is odd.
    # The number of ways is Product(C_{(L_i - 1) / 2}) where C_k is the k-th Catalan number.
    # Wait, the last block L_M doesn't have to be odd? 
    # No, the operation replaces l+1 ... r-1. The values at l and r remain.
    # To turn an alternating sequence of length L into a single value, L must be odd.
    # If L is even, you can't turn it into a single value using this operation.
    
    # Let's check Sample 1: N=6, A=1 1 1 1 1 0. 
    # Blocks: (1, 5), (0, 1). L_1=5, L_2=1.
    # C_{(5-1)/2} * C_{(1-1)/2} = C_2 * C_0 = 2 * 1 = 2.
    # Sample 1 output is 3. Why?
    # Because the last block (0) could have been part of a larger alternating 
    # sequence that was then reduced.
    # Actually, the constraint is: we can pick any l, r.
    # Let's re-evaluate. We have a sequence of blocks. 
    # An operation takes three consecutive blocks (X, Y, X) and turns them into one (X).
    # This is exactly the process of reducing a string via the rule "XYX -> X".
    # The number of ways to reduce a string of length N to a string of length M 
    # is the product of C_{(L_i - 1)/2} ONLY if we can't "overlap" operations.
    # But we can. However, the blocks are independent.
    # The only catch is the boundaries.
    # If we have blocks of lengths L_1, L_2, ..., L_M.
    # The total number of operations is (N - M) // 2.
    # The number of ways is (N-M)//2 ! / (Product (L_i-1)//2 !) * Product (C_{(L_i-1)//2})
    # No, that's for independent sets.
    # The correct formula for this specific problem is:
    # Ans = ( (N-M)//2 )! / Product( ((L_i-1)//2)! * ((L_i+1)//2)! )
    # This is equivalent to: ( (N-M)//2 )! * Product( 1 / ( ((L_i-1)//2)! * ((L_i+1)//2)! ) )
    # Let's test Sample 1: N=6, A=111110. L=[5, 1]. M=2. (6-2)//2 = 2.
    # 2! / ( (2! * 3!) * (0! * 1!) ) = 2 / (2 * 6 * 1) = 1/6. Not 3.
    
    # Let's use the property: the number of ways to reduce a sequence of length 2k+1 
    # to 1 is C_k. The total number of ways to reduce N to M is:
    # (Total Ops)! / Product( (Ops_i)! ) * Product( C_{Ops_i} )
    # where Ops_i = (L_i - 1) // 2.
    # Total Ops = (N - M) // 2.
    # Ans = ((N-M)//2)! / Product( (Ops_i)! ) * Product( (2*Ops_i)! / (Ops_i! * (Ops_i+1)!) )
    # Ans = ((N-M)//2)! * Product( (2*Ops_i)! / ( (Ops_i!)^2 * (Ops_i+1) ) )
    
    # Sample 1: N=6, A=111110. L=[5, 1]. M=2. Ops=[2, 0]. Total=2.
    # 2! / (2! * 0!) * (C_2 * C_0) = 1 * (2 * 1) = 2. Still not 3.
    # Wait, the sample says 3. Let's re-read.
    # (1,0,1,0,1,0) -> (1,0,0,0,1,0) [l=2, r=4] -> (1,1,1,1,1,0) [l=1, r=5]
    # (1,0,1,0,1,0) -> (1,1,1,1,1,0) [l=1, r=5] - No, l=1, r=5 means A[1]=1, A[5]=1, A[2,3,4] become 1.
    # Initial: X1=1, X2=0, X3=1, X4=0, X5=1, X6=0.
    # Op 1: l=1, r=3 -> X2=1. X=(1,1,1,0,1,0)
    # Op 2: l=3, r=5 -> X4=1. X=(1,1,1,1,1,0)
    # Or Op 1: l=3, r=5 -> X4=1. X=(1,0,1,1,1,0)
    # Op 2: l=1, r=3 -> X2=1. X=(1,1,1,1,1,0)
    # Or Op 1: l=1, r=5 -> X2,X3,X4=1. X=(1,1,1,1,1,0)
    # Total 3. This is exactly C_2 = 2? No, C_2 is 2. 0, 1, 1, 2, 5...
    # C_0=1, C_1=1, C_2=2, C_3=5.
    # For L=5, the number of ways to reduce to 1 is C_{(5-1)//2} = C_2 = 2.
    # But we also have the operation l=1, r=5 which is 1 op.
    # The number of ways to reduce a block of length 2k+1 to 1 is actually the 
    # number of binary trees with k internal nodes, but the operations can be 
    # performed in any order.
    # The number of ways to reduce a sequence of length 2k+1 to 1 is k! * C_k / (k+1)? No.
    # It's actually k! * C_k / (k+1) is not it.
    # The number of ways is k! * C_k / (something).
    # Let's use the formula: the number of ways to reduce a block of length 2k+1 to 1 is k!.
    # No, for k=2 (L=5), 2! = 2. But we found 3.
    # 3 is the 3rd Motzkin number? No.
    # 3 is (2k)! / (k! * (k+1)!) * k! ? No.
    # Actually, the number of ways to reduce a block of length 2k+1 to 1 is the 
    # number of ways to parenthesize the expression, which is C_k, 
    # and then multiply by the number of orderings of these operations, which is k!.
    # But some orderings are identical.
    # The correct answer for a block of length 2k+1 is k! * C_k / (k+1) ... no.
    # Let's use the known result for this problem: 
    # The number of ways to reduce a block of length 2k+1 to 1 is k!.
    # Wait, for k=2, 2! = 2. Still not