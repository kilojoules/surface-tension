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
    # The target state A is reachable if and only if it can be formed by 
    # repeatedly replacing "010" with "000" or "101" with "111".
    # This is equivalent to saying that A must be obtainable by taking the 
    # initial sequence (1, 0, 1, 0, ...) and collapsing contiguous blocks.
    
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # The initial sequence is X = [1, 0, 1, 0, 1, 0, ...]
    # The number of operations to reach A is (N - number of blocks in A).
    # However, we need to count the number of sequences of operations.
    
    # Key Insight:
    # This problem can be modeled as a tree structure of operations.
    # Each operation removes a "hump" (a block of different values).
    # The number of ways to reduce the sequence is related to the number of 
    # ways to linearize the tree of operations.
    # Specifically, if we have a sequence of blocks of lengths L1, L2, ..., Lk,
    # the number of ways is Product( (L_i + 1) ) for i=1 to k-1, but we must 
    # account for the specific constraints.
    
    # Let's re-evaluate:
    # The operation replaces a segment of length (r-l-1) of value Y with value X.
    # This is only possible if the segment was originally a sequence of alternating values
    # that had been reduced to a single block of value Y.
    # The total number of operations is exactly (N - (number of blocks in A)).
    # Let the lengths of blocks in A be b1, b2, ..., bk.
    # The number of ways is (b1 * b2 * ... * bk) / (some factor)? No.
    # Correct combinatorial result for this specific problem:
    # The answer is the product of (length of block i) for i = 1 to k-1, 
    # but only for blocks that "could have been" the middle of an operation.
    # Actually, the formula is: Product_{i=2}^{k-1} (length of block i + 1) 
    # if the sequence is valid, but we must check if A is reachable.
    
    # Check reachability:
    # A is reachable if A_i can be produced from X_i = i % 2.
    # This means we cannot have two adjacent blocks of the same value 
    # that didn't originate from the same X_i. But blocks are identical values,
    # so we just need to check if A can be formed.
    # Since we can only change X_i to X_{i-1} or X_{i+1}, 
    # the only way to get a block of length L is to absorb (L-1) elements.
    # The condition for reachability is that we never "lose" the ability to 
    # match the alternating pattern. Actually, any A is reachable if 
    # A_i != A_{i+1} is not violated in a way that prevents the operation.
    # But the operation requires l+1 < r, meaning the middle block must be at least 1.
    # The simplest check: A is reachable if we can reduce X to A.
    # X = 1 0 1 0 1 0...
    # If A = 1 1 0, we can take l=1, r=3 (X1=1, X3=1, X2=0) -> 1 1 1. Not 1 1 0.
    # Wait, the sample 1: X = 1 0 1 0 1 0, A = 1 1 1 1 1 0.
    # Op 1: l=2, r=4 -> X = 1 0 0 0 1 0.
    # Op 2: l=1, r=5 -> X = 1 1 1 1 1 0.
    
    # Let's use the property: the number of ways is the product of 
    # (length of block i) for i = 2, 4, 6... (the "middle" blocks).
    # No, the correct logic for this problem is:
    # 1. Compress A into blocks of (value, length).
    # 2. If any block i (1 < i < k) has length 0 (not possible) or if 
    #    the alternating property is violated, it's 0.
    # 3. The answer is Product_{i=2}^{k-1} (length of block i + 1).
    # Wait, let's test Sample 1: A = [1, 1, 1, 1, 1, 0]. Blocks: (1, 5), (0, 1).
    # k = 2. Product is empty = 1? Sample 1 says 3.
    # Let's re-read: X = 1 0 1 0 1 0. A = 1 1 1 1 1 0.
    # To get A, we need to eliminate X_2=0 and X_4=0.
    # X_2 can be eliminated by l=1, r=3. X_4 can be eliminated by l=3, r=5.
    # If we do (1,3) then (3,5), or (3,5) then (1,3), or (1,5) then (2,4)...
    # Actually, the number of ways to clear a segment of length L of opposite values
    # is the Catalan-like number or related to the number of binary trees.
    # For a block of length L in A that replaced a sequence of length 2L-1 in X,
    # the number of ways is the (L-1)-th Catalan number? No.
    
    # Correct Logic:
    # Each block in A of length L corresponds to a range in X.
    # If the block is at index i (1-indexed) and has length L,
    # it covers a range of the original X.
    # The number of ways to form a block of length L using the described operation
    # is given by the formula: (L+1)^(L-1) is for trees, but here it's simpler.
    # For a block of length L, the number of ways to "fill" it is 1 if L=1, 
    # and if L > 1, it depends on how many 0/1s were removed.
    # The number of ways to reduce a sequence of length 2m-1 to a single value 
    # is m! * (m-1)! / 2^(m-1)? No.
    
    # Let's use the known result for this specific problem:
    # The answer is Product_{i=1}^{k} (length of block i)! / (something)
    # Actually, the number of ways to form a block of length L is L!.
    # No, the formula is: Product_{i=1}^{k} (length of block i)! / 2^(length of block i - 1)
    # But we must divide by 2 for each block except the first and last?
    # Let's try: Sample 1: L1=5, L2=1. 5! / 2^4 = 120 / 16 = 7.5. Not 3.
    
    # Let's reconsider: to get a block of length L, we must have performed L-1 operations.
    # Each operation takes a 010 -> 000 or 101 -> 111.
    # This is like building a binary tree. The number of ways to build a 
    # binary tree with L leaves is the Catalan number C_{L-1}.
    # But the operations are ordered. The number of linear extensions of the 
    # poset of operations is (2L-2)! / 2^{L-1}.
    # For Sample 1: L1=5. (2*5-2)! / 2^(5-1) = 8! / 16 = 40320 / 16 = 2520. Still not 3.
    
    # Wait, the operation is: replace l+1 ... r-1 with X[l].
    # This means we remove a contiguous segment of the opposite value.
    # To get a block of length L, we must have removed L-1 blocks of the opposite value.
    # Each such block had length 1.
    # To remove L-1 blocks, we can pick any of the L-1 blocks and remove it, 
    # provided it is surrounded by the target value.
    # This is exactly the number of ways to empty L-1 boxes arranged in a line,
    # where you can only empty a box if its neighbors are already empty? No.
    # You can empty box i if box i-1 and box i+1 are the same value.
    # Initially: 1 0 1 0 1 0 1 0...
    # To remove X_2, we need X_1 and X_3. To remove X_4, we need X_3 and X_5.
    # This is like a game where you remove internal nodes.
    # The number of ways to remove L-1 elements is (L)! / 2^(L-1) ? No.
    # For L=2, ways=1. For L=3, ways=2. For L=4, ways=5. For L=5, ways=14.
    # These are Catalan numbers C_{L-1}.
    # Sample 1: L1=5, L2=1. C_{5-1} = C_4 = 14. Sample 1 says 3.
    # Let me re-read: "l+1 < r".
    # Sample 1: X = 1 0 1 0 1 0. A = 1 1 1 1 1 0.
    # We need to remove X_2 and X_4.
    # Op 1: l=1, r=3 (X_2 becomes 1). X = 1 1 1 0 1 0.
    # Op 2: l=3, r=5 (X_4 becomes 1). X = 1 1 1 1 1 0.
    # OR
    # Op 1: l=3, r=5 (X_4 becomes 1). X = 1 0 1 1 1 0.
    # Op 2: l=1, r=3 (X_2 becomes 1). X = 1 1 1 1 1 0.
    # OR
    # Op 1: l=1, r=5 (X_2, X_3, X_4 become 1). X = 1 1 1 1 1 0.
    # Total = 3.
    # For L=2, ways=1. For L=3, ways=3. For L=4, ways=10?
    # The number of ways to reduce a sequence of length 2L-1 to L is the 
    # number of binary trees with L leaves, but the operations are ordered.
    # This is known as the number of ways to parenthesize a product, 
    # but since we can pick any "peak" to flatten, it's different.
    # The formula for this specific problem is:
    # The number of ways to clear L-1 "humps" is (L+1)^{L-2} * (something)?
    # No, the formula is: for each block of length L, the number of ways is L! / 2^{L-1} 
    # is for a different problem.
    # Let's look at L=1 -> 1, L=2 -> 1, L=3 -> 3, L=4 -> 13, L=5 -> 75...
    # These are the Ordered Bell numbers? No.
    # Let's try: f(L) = \sum_{i=1}^{L-1} f(i) * f(L-i) * C(L-1, i)? No.
    # The correct recurrence is: