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

    # The operation is: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[i] becomes X[l] for l < i < r.
    # This is essentially merging a block of opposite values into the surrounding value.
    # The target state A is reachable if it can be formed by repeatedly 
    # replacing "010" with "000" or "101" with "111".
    # This is equivalent to saying that A must be obtainable by taking the 
    # alternating sequence (1, 0, 1, 0, ...) and performing these merges.
    # Crucially, the operation reduces the number of contiguous blocks of identical values.
    # Let the initial sequence be S. S has N blocks of size 1.
    # The operation takes three blocks (e.g., 0, 1, 0) and turns them into one block (0, 0, 0).
    # This reduces the number of blocks by 2.
    
    # First, check if A is reachable.
    # A is reachable if it can be reduced to the alternating sequence by "splitting" blocks.
    # Actually, a simpler way: A is reachable if we can partition A into blocks of 
    # identical values, and the number of blocks is consistent with the 
    # alternating start and the operations.
    
    # Let's compress A into blocks of (value, length).
    blocks = []
    if N > 0:
        current_val = A[0]
        count = 0
        for v in A:
            if v == current_val:
                count += 1
            else:
                blocks.append((current_val, count))
                current_val = v
                count = 1
        blocks.append((current_val, count))

    # The initial sequence is X_i = i % 2. 
    # Note: The problem says cell i (1 <= i <= N) has i % 2.
    # So X = [1, 0, 1, 0, 1, 0, ...]
    # The number of blocks in A must be reachable from N blocks.
    # Each operation reduces the number of blocks by 2.
    # Also, the sequence of values in blocks must be a subsequence of (1, 0, 1, 0, ...)
    # and must preserve the alternating property.
    
    # Validation:
    # The i-th block in A must have value (i % 2) if we index blocks from 1.
    # Wait, the first block of A must be 1 or 0. 
    # If the first block is 0, it means the first '1' was swallowed.
    # But the operation requires l and r to be the same. 
    # To turn X_1=1 into 0, we need X_0 and X_2 to be 0. But indices are 1 to N.
    # So X_1 can only change if there is some l < 1, which is impossible.
    # Therefore, A_1 must be equal to X_1 = 1 % 2 = 1.
    # Similarly, A_N must be equal to X_N = N % 2.
    
    if A[0] != 1 or A[-1] != (N % 2):
        print(0)
        return

    # Let k be the number of blocks in A.
    # The blocks in A must be (1, 0, 1, 0, ...).
    # If A[0] is 1, then block i (0-indexed) must have value (i + 1) % 2.
    for i in range(len(blocks)):
        if blocks[i][0] != (i + 1) % 2:
            print(0)
            return

    # Now we use DP. Let f(n, k) be the number of ways to form k blocks from n cells.
    # This is a known problem related to "staircase" structures or specific 
    # combinations. The number of ways to form the target sequence A is the 
    # product of (count_i + 1) / 2 rounded? No.
    # Let's use the property: to get a block of length L, we need to perform 
    # (L-1)//2 operations if the block is "internal".
    # The number of ways to form a block of length L from alternating bits 
    # using the described operation is the Catalan-like number.
    # Specifically, for a block of length L, the number of ways is 
    # the number of binary trees with (L-1)//2 internal nodes, which is 
    # C_{(L-1)//2} if L is odd, and 0 if L is even? 
    # No, the operation is: choose l, r such that X_l == X_r and X_{l+1...r-1} != X_l.
    # This means we can only merge a block of length 1 into two blocks of the same value.
    # To get a block of length L, we must have started with L alternating bits.
    # The only way to get a block of length L > 1 is to repeatedly merge 
    # the middle element.
    # For L=3: (1,0,1) -> (1,1,1). 1 way.
    # For L=5: (1,0,1,0,1) -> (1,1,1,0,1) -> (1,1,1,1,1) OR (1,0,1,1,1) -> (1,1,1,1,1). 2 ways.
    # This is exactly the Catalan number C_m where m = (L-1)//2.
    # But we can only merge if L is odd. If L is even, it's impossible?
    # Let's check: (1,0) cannot be merged. (1,0,1,0) cannot be merged into (1,1,1,1).
    # If we have (1,0,1,0), we can make it (1,1,1,0) or (1,0,0,0).
    # Both result in blocks of size 3 and 1.
    # The constraint is: A block of length L can be formed if and only if 
    # it consists of the same value and was derived from alternating values.
    # The number of ways to form a block of length L is C_{(L-1)//2} if L is odd, 
    # and 0 if L is even.
    # Wait, Sample 1: A = (1, 1, 1, 1, 1, 0). Blocks: (1, 5), (0, 1).
    # L=5 is odd, (5-1)//2 = 2. C_2 = 2.
    # L=1 is odd, (1-1)//2 = 0. C_0 = 1.
    # Total = 2 * 1 = 2? Sample 1 output is 3.
    # Let's re-read: "Two sequences of operations are different if lengths differ or (l, r) differ."
    # The operations can overlap.
    # Let's use the property: the number of ways to reduce a segment of length L 
    # to a single value is the number of binary trees with (L-1)//2 nodes, 
    # but the operations can be done in any order.
    # The number of ways to linearize a binary tree is (2m)! / (m+1)! / m! * m! ? No.
    # Actually, for a block of length L (L odd), the number of ways is (L)! / (2^((L-1)//2) * ((L-1)//2 + 1)!)
    # No, that's not it. Let's use the formula for the number of ways to 
    # reduce a string of length 2m+1 to 1 via this operation:
    # It is m! * C_m = (2m)! / (m+1)!.
    # For L=5, m=2, (4!)/3! = 4. For L=3, m=1, (2!)/2! = 1. For L=1, m=0, 1.
    # Sample 1: L=5, L=1 -> 4 * 1 = 4. Still not 3.
    # Let's re-evaluate. The operations are: choose l, r such that X_l == X_r and X_{l+1...r-1} != X_l.
    # This means r-l must be 2. The operation is: X_{l+1} becomes X_l.
    # Now we have X_l, X_l, X_r. Since X_l == X_r, we have a block of 3.
    # To get a block of length 5: (1,0,1,0,1)
    # 1. l=1, r=3 -> (1,1,1,0,1) -> l=3, r=5 -> (1,1,1,1,1)
    # 2. l=3, r=5 -> (1,0,1,1,1) -> l=1, r=3 -> (1,1,1,1,1)
    # 3. l=1, r=5 -> (1,1,1,1,1) - This is allowed because X_2, X_3, X_4 are not X_1?
    # No, "The integer written in cell i (l < i < r) is different from the integer written in cell l."
    # For l=1, r=5, X_2=0, X_3=1, X_4=0. But X_3 is 1, which is equal to X_1.
    # So l=1, r=5 is NOT allowed initially.
    # We must first clear X_3.
    # Sequence 1: (l=1, r=3), then (l=1, r=5) or (l=3, r=5).
    # Sequence 2: (l=3, r=5), then (l=1, r=3) or (l=1, r=5).
    # Total ways for L=5: 
    # Op 1: (1,3), Op 2: (1,5)
    # Op 1: (1,3), Op 2: (3,5)
    # Op 1: (3,5), Op 2: (1,3)
    # Op 1: (3,5), Op 2: (1,5)
    # Wait, if we do (1,3), the string becomes (1,1,1,0,1). Now l=1, r=5 satisfies the condition 
    # because X_2=1, X_3=1, X_4=0. NO, X_2 and X_3 are 1, which is X_1.
    # The condition is: X_i != X_l for ALL l < i < r.
    # So if we have (1,1,1,0,1), we cannot pick l=1, r=5 because X_2=1.
    # We must pick l=3, r=5.
    # Correct logic: To merge a block of length L, we must merge the "inner" 
    # alternating blocks first.
    # This is exactly the structure of a binary tree. 
    # A block of length 2m+1 is formed by merging two blocks of length 2k+1 and 2(m-k)+1 
    # with a 0-length (or rather, a single element) bridge? No.
    # Let's use the property: the number of ways to form a block of length L is 
    # the number of ways to parenthesize a product of m+1 elements, which is C_m, 
    # multiplied by the number of ways to order those operations.
    # For a binary tree, the number of linear extensions of the tree poset is 
    # (2m)! / Product(subtree_sizes).
    # But here, the operations are: to merge l and r, the middle must be different.
    # This means we can only merge l and r if the middle is a single block of the opposite color.
    # So we have a sequence of blocks: B1, B2, B3, ..., Bk.
    # We can merge B_i, B_{i+1}, B_{i+2} if B_i and B_{i+2} have the same color.
    # This is exactly the process of reducing a string by deleting "010