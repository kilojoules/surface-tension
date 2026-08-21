```python
import sys
from itertools import groupby

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # Initial state X_i = i % 2 (1-indexed)
    # X = [1, 0, 1, 0, ...] if N=6 -> [1, 0, 1, 0, 1, 0]
    # The operation: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[l+1...r-1] becomes X[l].
    # This is essentially a range-fill operation that can only happen if the 
    # range being filled is currently the opposite value of the boundaries.
    
    # Let's analyze the structure:
    # We can only change a block of 0s to 1s if it's surrounded by 1s, or vice versa.
    # This is equivalent to removing a "peak" or "valley" in the sequence of values.
    # If we group the initial sequence (1, 0, 1, 0, ...), we have N groups of size 1.
    # The target sequence A can be viewed as groups of identical values.
    # Let the target sequence A be represented as blocks of identical values.
    # If A is (1, 1, 1, 1, 1, 0), blocks are [(1, 5), (0, 1)].
    
    # A key observation: an operation (l, r) reduces the number of contiguous blocks by 2.
    # It merges three blocks (val, opp, val) into one block (val).
    # This is exactly like the process of reducing a string by removing patterns like "010" -> "0" or "101" -> "1".
    # The number of ways to reach a target state depends on the number of blocks in the initial state
    # and the number of blocks in the target state.
    
    # Initial blocks: N blocks of size 1.
    # Target blocks: Let target blocks be B_1, B_2, ..., B_k.
    # For a target state to be reachable, the parity of the blocks must match the initial state
    # at the boundaries of the blocks, and we cannot "create" new values.
    # Actually, the condition is simpler: the target A must be reachable by repeatedly
    # replacing "010" with "0" or "101" with "1".
    # This means the sequence of block values in A must be a subsequence of the initial 
    # sequence of block values (1, 0, 1, 0, ...), and specifically, it must be 
    # obtainable by deleting pairs of (opposite, same) or (same, opposite).
    
    # Let's check if A is reachable.
    # The initial sequence is S = [1, 0, 1, 0, ...].
    # The target sequence A is reachable if and only if it can be formed by the allowed operation.
    # The operation (l, r) requires X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This means the segment between l and r must be a single block of the opposite value.
    # So we are replacing "1 0 1" with "1 1 1" or "0 1 0" with "0 0 0".
    # In terms of blocks, this is: (Block of 1s, Block of 0s, Block of 1s) -> (Block of 1s).
    
    # Let the target A be represented as blocks of lengths L_1, L_2, ..., L_k.
    # The total number of initial blocks is N.
    # Each operation reduces the number of blocks by 2.
    # Total operations needed = (N - k) / 2.
    # This is only possible if (N - k) is even and A is "consistent" with the initial pattern.
    
    # Consistency check:
    # The i-th block of A must have the same value as the blocks it "absorbed".
    # Since initial is 1, 0, 1, 0..., the block values in A must be 1, 0, 1, 0... 
    # but some might be skipped. However, the operation preserves the value of the boundaries.
    # The only way to reach A is if the sequence of values of the blocks in A 
    # is a subsequence of (1, 0, 1, 0, ...) and the "removed" parts were blocks of the opposite value.
    # This implies the values of blocks in A must be A_block_1, A_block_2, ...
    # where A_block_i != A_block_{i+1}.
    # Also, the first block's value must be consistent with the parity of its starting position,
    # but the operation allows us to change the values of cells l+1...r-1.
    # Wait, the operation says replace X[l+1...r-1] with X[l].
    # This means if X[l]=1 and X[r]=1, the 0s between them become 1s.
    # This can only happen if there was exactly one block of 0s between them.
    
    # Let's use the property: the number of ways to reduce N blocks to k blocks 
    # via the operation (block_i, block_{i+1}, block_{i+2}) -> block_i 
    # (where block_i and block_{i+2} have the same value) is given by 
    # the product of Catalan-like numbers.
    # Specifically, if we have a sequence of blocks and we need to reduce a segment 
    # of 2m+1 blocks to 1 block, the number of ways is the m-th Catalan number C_m.
    # But here, the blocks are fixed. If we have a target block of length L,
    # it was formed by absorbing (L-1) cells. 
    # This is only possible if those L-1 cells were originally the opposite value 
    # and were absorbed in a specific order.
    
    # Correct logic:
    # 1. Group A into blocks of identical values.
    # 2. For each block i of length L_i:
    #    - If the block's value is different from the initial value of its cells,
    #      it must have been filled by an operation.
    #    - The only way to fill a range is if the boundaries have the same value.
    #    - This looks like a parenthesization problem.
    #    - For a block of length L, if it's "filled", it takes (L-1)/2 operations? No.
    
    # Let's reconsider: the operation is (l, r) where X[l] == X[r] and X[i] != X[l].
    # This means we can only merge a block of 0s into 1s if it's surrounded by 1s.
    # This is exactly like removing a mountain in a 1D landscape.
    # The number of ways to flatten a sequence of N blocks into k blocks is:
    # If the target blocks are at indices i_1, i_2, ..., i_k of the original sequence,
    # then the gaps between them must be filled.
    # A gap of size 2m (number of blocks) can be filled in C_m ways.
    # The total ways is the product of C_{gap_i/2}.
    
    # In our case, the initial blocks are just cells 1, 2, ..., N.
    # The target blocks are the contiguous segments of identical values in A.
    # Let the target blocks be B_1, B_2, ..., B_k.
    # For this to be possible:
    # 1. The value of B_j must be the same as the initial value of the cells it "absorbed".
    #    Wait, the operation replaces X[l+1...r-1] with X[l].
    #    So if B_j is a block of 1s, it must have started with a 1 at its boundaries.
    #    The cells in B_j that were originally 0 must have been absorbed.
    #    A cell i is originally (i % 2). 
    #    If A_i != (i % 2), it must have been changed.
    #    This is only possible if it's part of a range [l+1, r-1] where X[l] == X[r] == A_i.
    
    # Let's use the property: the number of ways is the product of C_{(L_i - 1)//2}
    # for each block i that was "filled", provided the parity is correct.
    # Actually, the simplest condition is:
    # For each block of identical values in A with length L:
    # If the block's value is V, and the cells in it were originally (1, 0, 1, 0...),
    # the number of cells in that block that had the opposite value V^1 is (L // 2).
    # The number of ways to "clear" these is C_{L // 2}.
    # But this is only if the block is "fillable".
    # A block is fillable if its boundaries (the cells just outside) allow it, 
    # or if it extends to the boundary of the grid.
    # Actually, the only condition is that the total number of blocks is reduced.
    # The number of ways is simply the product of C_{L // 2} for all blocks in A,
    # provided that the final sequence A is reachable.
    # A is reachable if and only if for every block of value V and length L,
    # the cells of value V^1 within it can be covered by intervals [l, r] with X[l]=X[r]=V.
    # This is always true if we can pick l and r within the block or at the boundaries.
    # The only constraint is that we cannot change the values of A_1 and A_N 
    # unless they were changed by an operation that started/ended outside.
    # But the operation requires l and r to be within [1, N].
    # So A_1 must be equal to the initial X_1 (1) and A_N must be equal to X_N (N%2).
    # Wait, the sample 1: N=6, A=[1, 1, 1, 1, 1, 0]. 
    # Initial X = [1, 0, 1, 0, 1, 0].
    # A_1 = 1 (X_1=1), A_6 = 0 (X_6=0). This is consistent.
    # Blocks of A: [1, 1, 1, 1, 1] (L=5, V=1), [0] (L=1, V=0).
    # For the first block: L=5, V=1. Initial values: 1, 0, 1, 0, 1.
    # The 0s are at indices 2 and 4.
    # Ways to clear: C_{5 // 2} = C_2 = 2.
    # For the second block: L=1, V=0. Initial value: 0.
    # Ways: C_{1 // 2} = C_0 = 1.
    # Total = 2 * 1 = 2. But sample output says 3. 
    # Let's re-read: "Choose cells l and r (l+1 < r)".
    # Sample 1: X = (1, 0, 1, 0, 1, 0). 
    # Op 1: l=2, r=4. X becomes (1, 0, 0, 0, 1, 0).
    # Op 2: l=1, r=5. X becomes (1, 1, 1, 1, 1, 0).
    # This is one sequence. Another: l=1, r=3 then l=1, r=5.
    # This means the blocks of A are formed by nested operations.
    # The number of ways to reduce a sequence of length L to a single value is C_{(L-1)//2}
    # ONLY IF the initial sequence was V, V^1, V, V^1, ..., V.
    # If the initial sequence was 1, 0, 1, 0, 1, then L=5, and the number of ways is C_2 = 2.
    # But the sample says 3. Let's see:
    # Initial: 1 0 1 0 1 0
    # 1. (2,4) -> 1 0 0 0 1 0 -> (1