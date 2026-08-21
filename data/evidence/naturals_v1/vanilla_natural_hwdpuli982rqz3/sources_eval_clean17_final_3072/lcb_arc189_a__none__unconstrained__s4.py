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
    # alternating sequence (1, 0, 1, 0...) and collapsing contiguous blocks.
    
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # The initial state is X = [1, 0, 1, 0, 1, 0, ...]
    # To reach A, we must have started with a sequence of blocks in X and 
    # merged them. Specifically, a block of length k in A corresponds to 
    # a sequence of blocks in X that were merged.
    # If a block in A has value v and length k, it must have come from 
    # a sequence in X starting and ending with v, with alternating values in between.
    # The number of ways to form a block of length k using these operations 
    # is known to be the (k-1)-th Catalan number if we consider the nesting 
    # structure of the operations, but the problem constraints and operation 
    # definition actually map to a simpler combinatorial structure.
    
    # Correct logic: 
    # 1. Check if A is reachable. A is reachable if it doesn't contain 
    #    patterns that cannot be formed. However, since we start with 1,0,1,0...
    #    any A is reachable as long as we don't try to "create" a value 
    #    where one didn't exist. But we can only replace l+1...r-1.
    #    Actually, the condition is: A must be obtainable by deleting 
    #    elements from the alternating sequence X such that we only 
    #    delete "middle" elements.
    #    Wait, the operation is: X[l+1...r-1] = X[l] if X[l] == X[r].
    #    This means we can eliminate a block of length 1 (or any odd length) 
    #    of the opposite color.
    
    # Let's analyze the blocks of A.
    # Let the blocks of A be B_1, B_2, ..., B_m with lengths L_1, L_2, ..., L_m.
    # For each block i, the number of ways to form it is C_{ (L_i - 1) // 2 }.
    # But we must also ensure that the alternating pattern is maintained.
    # If A_i != i % 2 (using 1-based), it means the parity shifted.
    # The only way to change the parity of the sequence is to perform an 
    # operation that removes an odd number of elements.
    # But the operation replaces X[l+1...r-1] with X[l]. 
    # The number of elements removed is (r-1) - (l+1) + 1 = r - l - 1.
    # For X[l] to equal X[r], r-l must be even. Thus r-l-1 is odd.
    # Every operation removes an odd number of elements.
    
    # Let's use the property: the number of ways to reduce a sequence of 
    # length k to a single block is the Catalan number C_{(k-1)//2} if k is odd, 
    # and 0 if k is even? No.
    # Actually, the number of ways to form a block of length L is 
    # the number of binary trees with (L-1)//2 internal nodes, which is C_{(L-1)//2}.
    # This is only possible if L has the same parity as the original block 
    # it replaced.
    
    # Let's re-evaluate:
    # The initial sequence is X_i = i % 2.
    # We can merge X[l...r] into X[l] if X[l] == X[r] and X[l+1...r-1] are all opposite.
    # This means we can replace a segment of length 3 (e.g., 1 0 1) with (1 1 1).
    # This operation preserves the values of X at indices l and r and changes 
    # the middle.
    # Crucially, this operation does not change the parity of the indices of the 
    # remaining "original" elements.
    
    # Let's look at the blocks of A. 
    # A block of length L consisting of value V starting at index i.
    # This block must have been formed from an original segment of X.
    # The only way to get a block of length L is if the original segment 
    # had length L, L+2, L+4... 
    # The number of ways to form a block of length L from a segment of length L + 2k 
    # is C_k.
    # But we don't know k. We know the total N.
    # This is a DP problem. 
    # Let dp[i] be the number of ways to form the prefix of A of length i.
    # To form A[i], we can take a segment of X of length 2k+1 and collapse it to 1.
    # This is getting complex. Let's use the property:
    # The answer is the product of C_{(L_i - 1) // 2} where L_i are lengths of 
    # blocks of identical elements in A, PROVIDED that A is reachable.
    # A is reachable if A_i = i % 2 for all i where A_i != A_{i-1}.
    # No, that's not quite right.
    
    # Correct Property:
    # The number of ways is the product of Catalan numbers C_{(L_i - 1) // 2} 
    # for each block i, but only if L_i is odd. If any L_i is even, the answer is 0.
    # Wait, Sample 1: A = [1, 1, 1, 1, 1, 0]. Blocks: (1, 5), (0, 1).
    # L_1 = 5, L_2 = 1. C_{(5-1)//2} = C_2 = 2. C_{(1-1)//2} = C_0 = 1. 2 * 1 = 2.
    # But sample output is 3. My block logic is wrong.
    
    # Let's use the property from similar problems:
    # The answer is the product of C_{L_i // 2} where L_i are the lengths of 
    # contiguous segments of the SAME value in A.
    # Sample 1: L = [5, 1]. C_{5//2} * C_{1//2} = C_2 * C_0 = 2 * 1 = 2. Still not 3.
    # Sample 1 again: A = [1, 1, 1, 1, 1, 0]. X = [1, 0, 1, 0, 1, 0].
    # We can get A by:
    # 1. (2, 4) -> X=[1, 0, 0, 0, 1, 0], then (1, 5) -> X=[1, 1, 1, 1, 1, 0]
    # 2. (4, 6) -> X=[1, 0, 1, 0, 0, 0], then (1, 5) -> X=[1, 1, 1, 1, 1, 0] (Wait, (1,5) is l=1, r=5, X[1]=1, X[5]=1, X[2,3,4]=0. Yes.)
    # 3. (2, 4) then (3, 5) -> X=[1, 0, 0, 0, 1, 0] then [1, 0, 1, 1, 1, 0]... no.
    # Actually, the 3 ways for Sample 1 are:
    # Op1: (2,4), Op2: (1,5)
    # Op1: (4,6), Op2: (1,5)
    # Op1: (1,5) - but X[2,3,4] are [0,1,0], not all same. So (1,5) cannot be first.
    # Wait, if we do (2,4), X becomes 1 0 0 0 1 0. Then (1,5) is valid.
    # If we do (4,6), X becomes 1 0 1 0 0 0. Then (1,5) is valid.
    # If we do (2,4) then (4,6), X becomes 1 0 0 0 0 0. Then (1,6) is not possible.
    
    # The actual pattern is: for each block of length L, the number of ways to 
    # form it is the number of ways to parenthesize the merges.
    # This is C_{ (L-1)//2 } if we only merge 3 into 1.
    # But we can merge any r-l-1.
    # The number of ways to form a block of length L is actually the 
    # number of binary trees where each node has 2 children, and the total 
    # number of leaves is (L+1)//2. This is C_{(L-1)//2}.
    # Let's re-read: "replace each of the integers written in cells l+1, ..., r-1 with X[l]".
    # This is exactly the process of building a binary tree.
    # For a block of length L, the number of ways is C_{(L-1)//2} if L is odd.
    # If L is even, it's 0? No, Sample 1 has L=5 and L=1. Both odd.
    # Sample 2: A = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0]. 
    # Blocks: L = [5, 1, 3, 1]. 
    # C_{(5-1)//2} * C_{(1-1)//2} * C_{(3-1)//2} * C_{(1-1)//2} = C_2 * C_0 * C_1 * C_0 = 2 * 1 * 1 * 1 = 2.
    # But sample output is 9.
    
    # Let's reconsider. The blocks of A are formed by taking the alternating 
    # sequence X and "filling in" the gaps.
    # To get a block of length L, we need a segment of X of length L, L+2, L+4...
    # Let k = (L-1)//2. The number of ways to form a block of length L 
    # using k operations is C_k.
    # But we can distribute these operations across different blocks.
    # Total operations = (N - (number of blocks)) // 2.
    # Let m be the number of blocks. Total ops = (N - m) // 2.
    # The number of ways is (Total Ops)! / (prod (k_i!)) * prod (C_{k_i})
    # where k_i is the number of operations used for block i.
    # k_i = (L_i - 1) // 2.
    # Total ways = ((N-m)//2)! / prod(k_i!) * prod(k_i! / (k_i+1)!) 
    # = ((N-m)//2)! / prod((k_i+1)!)
    # No, that's not it. The operations are ordered.
    # The correct formula for this specific problem is:
    # Ans = ((N-m)//2)! / prod( (k_i)! ) * prod( (k_i + 1) ) / (something)
    # Actually, the number of ways to form each block independently is C_{k_i}.
    # Since the operations for different blocks are independent and can be 
    # interleaved in any order, the total number of ways