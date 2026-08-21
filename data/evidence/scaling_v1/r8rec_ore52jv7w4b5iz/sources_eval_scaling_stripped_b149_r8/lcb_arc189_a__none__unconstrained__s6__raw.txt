```python
import sys
from itertools import groupby

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    mod = 998244353

    # Initial state: X_i = i % 2 (1-indexed)
    # X = [1, 0, 1, 0, 1, 0, ...]
    # The operation: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[i] becomes X[l] for all l < i < r.
    # This is essentially a range-fill operation that can only happen if the 
    # range is currently filled with the opposite value.
    # This structure is identical to the process of collapsing blocks in a 
    # binary string. Specifically, it maps to the number of ways to build 
    # a target binary string using a stack-like operation or 
    # counting valid bracket sequences/trees.
    
    # The problem can be modeled as: we have blocks of identical characters.
    # An operation reduces the number of blocks.
    # If we have a sequence of blocks like ... 0 1 0 ... 
    # we can merge the 0s and turn the 1 into a 0.
    # This is only possible if the target A allows it.
    
    # Let's group the target A into blocks of identical values.
    # groups = [(value, length), ...]
    groups = [(k, len(list(g))) for k, g in groupby(a)]
    m = len(groups)
    
    # The initial sequence is 1, 0, 1, 0...
    # The only way to reach A is if A is "consistent" with the initial parity.
    # However, the operation allows changing values.
    # The core constraint is that we can only overwrite a block of different 
    # values if it is surrounded by the same value.
    # This is equivalent to saying we are removing "peaks" or "valleys" in a 
    # 1D landscape.
    
    # For a target sequence A, the number of ways to reach it is related to 
    # the number of ways to parenthesize the reductions.
    # If we have m blocks, and we need to reduce the initial N blocks to m blocks,
    # the number of operations is (InitialBlocks - m) / 2 ? No.
    
    # Let's analyze the structure:
    # Each operation reduces the number of contiguous blocks by 2.
    # Initial blocks: N. Target blocks: m.
    # Total operations needed: (N - m) / 2.
    # This is only possible if N and m have the same parity and N >= m.
    # Also, the blocks must match the initial parity pattern.
    # Actually, the only restriction is that we cannot change the values of 
    # the very first and very last cells if they don't match the initial.
    # Wait, the initial values are X_i = i % 2.
    # X_1 = 1, X_2 = 0, X_3 = 1...
    
    # Check if A is reachable:
    # The operation preserves the values at the boundaries of the range [l, r].
    # It is known that the number of ways to reduce a sequence of blocks 
    # via this specific operation is given by the product of 
    # Catalan-like numbers or combinations.
    # Specifically, for each block of length L in the target A, 
    # if it was formed by merging, it corresponds to a binary tree.
    # The number of ways to merge k blocks into 1 is the (k-1)-th Catalan number.
    # But here, we merge blocks of alternating values.
    
    # Correct logic:
    # The target A is reachable if and only if:
    # 1. A[0] == 1 (since X_1 = 1)
    # 2. A[N-1] == (N % 2) (since X_N = N % 2)
    # 3. We can reach A from X.
    
    # Let's refine: the operation is essentially removing a block of 
    # length 1 (or more) that is surrounded by blocks of the opposite value.
    # This is like the game "Zuma" or removing matched parentheses.
    # The number of ways to reduce a sequence of blocks is the product of 
    # combinations. For a block of length L in the target, it "covers" 
    # some number of initial blocks.
    
    # Let's use the property: the number of ways is the product of 
    # C(length_of_block + 1, 2) is not correct.
    # The actual formula for this specific problem is:
    # If A is reachable, the answer is the product of (L_i * (L_i + 1) // 2) 
    # for all blocks i that were "expanded", but that's for a different problem.
    
    # Re-evaluating: the operation is: [0, 1, 0] -> [0, 0, 0].
    # This reduces the number of blocks by 2.
    # To get from N blocks to m blocks, we need (N-m)//2 operations.
    # Each operation removes one block of the "opposite" value.
    # The number of ways to do this is the product of (len(block) + 1) 
    # for the blocks that "absorbed" others? No.
    
    # Let's look at the sample: N=6, A=[1,1,1,1,1,0].
    # Initial: [1, 0, 1, 0, 1, 0]. Blocks: 6. Target blocks: 2.
    # Ops: (6-2)//2 = 2.
    # Blocks of A: (1, 5), (0, 1).
    # The first block of 1s has length 5. It absorbed two 0s.
    # The number of ways to absorb k blocks is the k-th Catalan number?
    # For k=2, C_2 = 2. But the answer is 3.
    # 3 is (k+1)! / (k! * 1!) ... no. 3 is Comb(2+1, 2) = 3.
    # For Sample 2: N=10, A=[1,1,1,1,1,0,1,1,1,0].
    # Blocks: (1, 5), (0, 1), (1, 3), (0, 1).
    # m = 4. Ops = (10-4)//2 = 3.
    # Block 1 (len 5) absorbed two 0s. Block 3 (len 3) absorbed one 0.
    # Ways = Comb(2+1, 2) * Comb(1+1, 1) = 3 * 2 = 6? Sample says 9.
    # Wait, the blocks absorbed are:
    # Block 1: absorbed X_2, X_4. (2 blocks)
    # Block 3: absorbed X_7, X_9? No, X_7 is 1.
    # Initial: 1 0 1 0 1 0 1 0 1 0
    # Target:  1 1 1 1 1 0 1 1 1 0
    # Indices: 1 2 3 4 5 6 7 8 9 10
    # Target blocks: [1,5], [6,6], [7,9], [10,10]
    # Block 1 (indices 1-5) absorbed X_2 and X_4.
    # Block 3 (indices 7-9) absorbed X_8.
    # Total ways = (Ways to absorb 2 blocks) * (Ways to absorb 1 block)
    # For k=2, ways=3. For k=1, ways=2. 3 * 2 = 6. Still not 9.
    
    # Let's re-read: "Two sequences are different if lengths differ or (l, r) differ."
    # The number of ways to reduce k blocks is (k+1)^(k-1) / k! ... no.
    # Actually, the number of ways to merge k blocks is (k+1)^{k-1} is for trees.
    # Let's try: for each block of length L in A, if it absorbed k blocks,
    # the number of ways is (k+1)^{k-1} ? No.
    # Let's try: for k=2, ways=3. For k=1, ways=3? No.
    # If k=1, ways=1. If k=2, ways=3. 3 * 1 = 3.
    # For Sample 2: Block 1 absorbed 2, Block 3 absorbed 1.
    # If k=1 -> 1 way, k=2 -> 3 ways. 3 * 1 = 3. Still not 9.
    
    # Wait! The number of ways to merge k blocks is (k+1)^{k-1} is for labeled trees.
    # What if the formula is (k+1)^{k-1} for k=1 is 1, k=2 is 3?
    # (1+1)^{1-1} = 2^0 = 1.
    # (2+1)^{2-1} = 3^1 = 3.
    # (3+1)^{3-1} = 4^2 = 16.
    # For Sample 2: Block 1 (k=2) -> 3 ways, Block 3 (k=1) -> 1 way. Total = 3.
    # Still not 9. Let me re-calculate k.
    # Sample 2: 1 1 1 1 1 0 1 1 1 0
    # Initial: 1 0 1 0 1 0 1 0 1 0
    # Block 1: indices 1-5. Initial values: 1 0 1 0 1. Absorbed two 0s. k=2.
    # Block 2: index 6. Initial: 0. Absorbed zero. k=0.
    # Block 3: indices 7-9. Initial: 1 0 1. Absorbed one 0. k=1.
    # Block 4: index 10. Initial: 0. Absorbed zero. k=0.
    # Maybe the formula is (k+1)^{k-1} is wrong.
    # What about (k+2)^{k-1}? 
    # k=1: 3^0 = 1.
    # k=2: 4^1 = 4.
    # 4 * 1 = 4.
    
    # Let's try another approach. The number of ways to reduce k blocks 
    # is given by the formula: (k+1)^{k-1} is for something else.
    # The number of ways to reduce k blocks is actually (k+1)^{k-1} 
    # ONLY if the blocks are distinct. Here they are identical.
    # The correct formula for this problem is: 
    # For each block of length L, let k be the number of blocks it absorbed.
    # The number of ways is (k+1)^{k-1} ? No.
    # Let's try: for k=1, ways=1; for k=2, ways=3; for k=3, ways=15?
    # Wait, Sample 2 answer is 9. 9 = 3 * 3.
    # That means both Block 1 (k=2) and Block 3 (k=1) contributed 3?
    # No, that doesn't make sense.
    # What if Block 1 (k=2) contributed 3 and Block 3 (k=1) contributed 3?
    # Or Block 1 contributed 9 and Block 3 contributed 1?
    # If k=1 gives 3 and k=2 gives 3, that's unlikely.
    # If k=1 gives 1 and k=2 gives 9?
    # Let's check k=2: (k+1)^{k-1} = 3^1 = 3.
    # If k=1 gives 3, then (1+1)^{1-1} = 1.
    # What if the formula is (k+2)^{k-1} ? k=1 -> 3^0 = 1, k=2 -> 4^1 = 4.
    # What if the formula is (k+2)^k / (k+1)? 
    # k=1: 3^1 / 2 = 1.5.
    # What