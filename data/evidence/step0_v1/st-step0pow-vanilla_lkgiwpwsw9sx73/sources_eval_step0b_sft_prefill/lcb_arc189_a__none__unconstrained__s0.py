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
    # and fill everything between them with that value, provided the middle
    # was different. This is essentially a way to merge blocks of identical values.
    # The target configuration A consists of several contiguous blocks of 0s and 1s.
    # Let the blocks be B_1, B_2, ..., B_k.
    # To reach A from the alternating sequence (1, 0, 1, 0...), we can only
    # expand a block if it "swallows" a block of the opposite color.
    # This structure is equivalent to counting the number of ways to build a 
    # rooted tree (or forest) where each node represents a block and edges 
    # represent the operations.
    
    # Specifically, if we have a sequence of blocks of lengths L_1, L_2, ..., L_k,
    # the number of ways to form this via the described operation is the product
    # of Catalan-like numbers. For a block of length L, the number of ways to 
    # form it by absorbing neighbors is given by the formula:
    # (2^(L-1)) if we consider the binary structure, but actually, 
    # the problem maps to: for each internal block i (1 < i < k), 
    # it must be absorbed by either block i-1 or block i+1.
    # This is a known problem where the answer is the product of 
    # (number of ways to reduce the sequence).
    
    # Let's refine: 
    # A block of length L consists of L identical elements.
    # To get a block of length L > 1, we must have performed operations.
    # The number of ways to form a block of length L using the described 
    # operation is (L-1)! * 2^(L-1) / L! ... no, that's not it.
    # Actually, for a contiguous segment of length L of the same character,
    # the number of ways to form it is the (L-1)-th Catalan number? No.
    # Let's re-evaluate: 
    # To form a block of length L, we need L-1 operations.
    # Each operation takes two indices l, r.
    # For a block of length L, the number of ways is 2^(L-1) if we can 
    # pick either the left or right boundary to expand.
    # But we must ensure we don't violate the "different" constraint.
    
    # Correct logic:
    # 1. Check if A is reachable. A is reachable if A_i = i % 2 for i=0 or i=N-1
    #    (depending on 0/1 indexing). Wait, the constraint is simpler:
    #    The operation doesn't change A_1 or A_N.
    #    Initial: X_i = i % 2 (1-indexed). So X_1 = 1, X_2 = 0, X_3 = 1...
    #    If A_1 != 1 or A_N != (N % 2), it's impossible? 
    #    No, the sample 1: N=6, A=(1,1,1,1,1,0). X=(1,0,1,0,1,0).
    #    A_1=1, X_1=1. A_6=0, X_6=0. This matches.
    
    # The number of ways to form a block of length L is (L+1)^(L-1) / L? No.
    # The actual combinatorial result for this specific problem is:
    # For each block of length L_i, the contribution is L_i.
    # The total number of ways is the product of L_i for all i such that 
    # the block is "internal" (not the first or last block), but that's for 
    # different constraints.
    
    # Re-evaluating: The operation is replacing [l+1, r-1] with X_l.
    # This is equivalent to deleting the blocks between l and r.
    # To get A, we start with N blocks of length 1.
    # We merge them. To merge 3 blocks (1, 0, 1) into (1, 1, 1), we use l=1, r=3.
    # This reduces the number of blocks by 2.
    # To end up with k blocks, we need (N - k) / 2 operations.
    # Each operation removes two blocks (one 0 and one 1).
    # For a final block of length L, it was formed by merging L blocks.
    # The number of ways to form a block of length L is L^(L-2) * (something)?
    # No, the formula for this specific problem is:
    # Answer = Product of (L_i) for all i, where L_i are lengths of contiguous blocks,
    # but only for blocks that were "expanded".
    # Actually, the simplest correct formula for this problem is:
    # Let the lengths of contiguous blocks in A be L_1, L_2, ..., L_k.
    # The answer is Product_{i=2}^{k-1} L_i.
    # Wait, Sample 1: A = (1, 1, 1, 1, 1, 0). Blocks: L_1=5, L_2=1.
    # k=2. Product is empty? That would be 1. But answer is 3.
    # Sample 2: A = (1, 1, 1, 1, 1, 0, 1, 1, 1, 0). Blocks: L_1=5, L_2=1, L_3=3, L_4=1.
    # k=4. L_2=1, L_3=3. Product = 1 * 3 = 3. But answer is 9.
    
    # Let's reconsider: each block i (1 < i < k) must be absorbed.
    # A block of length L_i can be absorbed in L_i ways?
    # No, the formula is: Answer = Product_{i=1}^{k} (L_i + 1) / 2 ... no.
    # The correct formula for this problem is:
    # For each block i from 1 to k, if i is even, it must be absorbed by 
    # either block i-1 or block i+1.
    # This is a DP. dp[i][0/1] = ways to process first i blocks.
    # But there is a known result: the answer is the product of L_i for all even i,
    # multiplied by the product of L_i for all odd i (excluding boundaries).
    # Let's test Sample 2: L = [5, 1, 3, 1]. Even indices: L_2=1, L_4=1. Odd: L_1=5, L_3=3.
    # Internal: L_2=1, L_3=3. 1 * 3 = 3. Still not 9.
    
    # Correct logic: Each internal block i (1 < i < k) must be absorbed.
    # It can be absorbed by the block to its left or the block to its right.
    # This creates a structure of dependencies.
    # The number of ways is Product_{i=2}^{k-1} (L_i + 1) if we can pick either side?
    # No. The actual formula is: Product_{i=2}^{k-1} (L_i + 1) is for a different problem.
    # For this one: each internal block i must be absorbed.
    # If i is even, it's absorbed by i-1 or i+1.
    # This is equivalent to: for each i from 2 to k-1, we have L_i ways to 
    # "center" the operation.
    # Actually, the answer is Product_{i=2}^{k-1} (L_i + 1) is almost it.
    # Let's try: Sample 1: L=[5, 1]. k=2. Product empty = 1. Still not 3.
    # Sample 2: L=[5, 1, 3, 1]. k=4. (L_2+1)(L_3+1) = (1+1)(3+1) = 8. Not 9.
    
    # One more try: The number of ways to form a block of length L is L.
    # For Sample 1: L=[5, 1]. The first block is length 5. It took 2 operations.
    # The operations were (2, 4) then (1, 5). Or (4, 6) then (1, 5)...
    # For a block of length L, the number of ways to form it is (L+1)//2 if L is odd?
    # No. The number of ways to form a block of length L is L if L is even, and (L+1)/2 if L is odd?
    # Let's use the property: an internal block of length L can be absorbed in L ways.
    # Sample 1: L=[5, 1]. Internal blocks: none. But the first block is length 5.
    # The first block is A_1...A_5. It was formed by absorbing A_2, A_3, A_4.
    # These were 3 blocks of length 1.
    # The number of ways to merge n blocks of length 1 into one block is n!.
    # But we have constraints.
    # The correct combinatorial formula for this problem is:
    # For each i from 1 to k, let L_i be the length of the i-th block.
    # The answer is Product_{i=1}^{k} (L_i + 1) // 2.
    # Sample 1: (5+1)//2 * (1+1)//2 = 3 * 1 = 3. Correct!
    # Sample 2: (5+1)//2 * (1+1)//2 * (3+1)//2 * (1+1)//2 = 3 * 1 * 2 * 1 = 6. Still not 9.
    
    # Wait, the Sample 2 answer is 9. (5+1)//2 = 3, (1+1)//2 = 1, (3+1)//2 = 2, (1+1)//2 = 1.
    # 3 * 1 * 2 * 1 = 6.
    # What if it's (L_i // 2) + 1?
    # Sample 1: (5//2 + 1) * (1//2 + 1) = 3 * 1 = 3.
    # Sample 2: (5//2 + 1) * (1//2 + 1) * (3//2 + 1) * (1//2 + 1) = 3 * 1 * 2 * 1 = 6.
    # What if it's (L_i + 1) // 2 but we only multiply for i < k?
    # Sample 2: 3 * 1 * 2 = 6.
    # What if the formula is Product (L_i + 1) // 2 but for Sample 2, L_3 is 3, so (3+1)//2 = 2.
    # Is it possible the blocks are merged differently?
    # Let's re-read: "Two sequences of operations are different if their lengths are different or (l, r) differ."
    # This is a known problem. The answer is Product_{i=1}^{k-1} (L_i + 1) // 2 * (L_{i+1} + 1) // 2 ... no.
    # The correct formula is: Answer = Product_{i=1}^{k} (L_i // 2 + 1) is wrong.
    # Let's try: Answer = Product_{i=1}^{k-1} (L_i // 2 + 1) * (L_{i+1} // 2 + 1) ... no.
    # Let's use the property: each block i (1 < i < k) must be absorbed.
    # To absorb a block of length L, there are (L+1)//2 ways.
    # For Sample 2: L=[5, 1, 3, 1]. Internal blocks are L_2=1, L_3=3.
    # Ways = (1+1)//2 * (3+1)//2 = 1 * 2 = 2.
    # But we also need to consider the boundaries.
    # The actual formula is: Answer = Product_{i=1