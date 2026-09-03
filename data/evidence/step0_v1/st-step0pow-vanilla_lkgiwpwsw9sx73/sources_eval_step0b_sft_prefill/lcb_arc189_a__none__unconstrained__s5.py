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
    # initial sequence (1, 0, 1, 0, ...) and collapsing contiguous segments.
    
    # Let's represent A as a sequence of blocks of identical consecutive elements.
    # E.g., 1 1 1 1 1 0 -> (1, 5), (0, 1)
    blocks = []
    if N > 0:
        current_val = A[0]
        count = 0
        for val in A:
            if val == current_val:
                count += 1
            else:
                blocks.append((current_val, count))
                current_val = val
                count = 1
        blocks.append((current_val, count))

    # The initial sequence is X_i = i % 2.
    # X = [1, 0, 1, 0, 1, 0, ...]
    # An operation (l, r) is valid if X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This means the segment [l+1, r-1] must consist of a single value different from X[l].
    # This operation effectively increases the length of a block of identical values.
    
    # Let the blocks of A be B_1, B_2, ..., B_k.
    # For A to be reachable, the sequence of values in B must be a subsequence of 
    # the initial alternating sequence (1, 0, 1, 0, ...), and specifically,
    # since we can only expand blocks, the "skeleton" of A must match the skeleton of X.
    # Actually, the condition is simpler: A is reachable if and only if 
    # A_i can be produced by the operations. 
    # The operation allows us to turn "010" -> "000" or "101" -> "111".
    # This means we can remove any internal block of length 1.
    # If we have a block of length > 1, it must have been created by an operation.
    # A block of length L > 1 requires (L-1) operations to be created if it absorbed 
    # blocks of length 1.
    
    # Let's use the property: the number of ways to form a block of length L 
    # from an alternating sequence using these operations is the Catalan-like 
    # number, specifically the number of binary trees.
    # For a block of length L, it takes L-1 operations. 
    # The number of ways to perform these operations is given by the 
    # (L-1)-th Catalan number? No, let's re-evaluate.
    
    # If we have a sequence of blocks with lengths l_1, l_2, ..., l_k.
    # The total number of operations is sum(l_i - 1) for l_i > 1? No.
    # Let's look at the structure: to get a block of length L, we must have 
    # started with alternating values. The only way to get L consecutive 1s 
    # is to take 1 0 1 0 1 ... 1 and collapse the 0s.
    # To get L consecutive 1s, we need L 1s and (L-1) 0s.
    # The number of ways to collapse (L-1) 0s into 1s is (L-1)! * (something)?
    # No, the operations must be nested or disjoint. This is exactly the 
    # structure of binary trees. The number of ways to reduce a sequence of 
    # length 2L-1 (1 0 1 0 1) to length L (1 1 1) is the Catalan number C_{L-1}.
    
    # Wait, the Sample 1: N=6, A=[1,1,1,1,1,0]. 
    # X = [1, 0, 1, 0, 1, 0]. 
    # To get A, we need to turn X[2]=0 into 1 and X[4]=0 into 1.
    # Op 1: l=2, r=4 -> X becomes [1, 0, 0, 0, 1, 0] (Incorrect, l+1 < r)
    # Let's re-read: l+1 < r. X_l == X_r, X_i != X_l for l < i < r.
    # Sample 1 again: X = (1, 0, 1, 0, 1, 0)
    # 1. l=2, r=4: X[2]=0, X[4]=0. X[3]=1. X[3] becomes 0. X = (1, 0, 0, 0, 1, 0)
    # 2. l=1, r=5: X[1]=1, X[5]=1. X[2,3,4]=0. X[2,3,4] become 1. X = (1, 1, 1, 1, 1, 0)
    # This is one way.
    # Another way:
    # 1. l=1, r=3: X[1]=1, X[3]=1. X[2]=0 -> 1. X = (1, 1, 1, 0, 1, 0)
    # 2. l=3, r=5: X[3]=1, X[5]=1. X[4]=0 -> 1. X = (1, 1, 1, 1, 1, 0)
    # Another way:
    # 1. l=3, r=5: X[3]=1, X[5]=1. X[4]=0 -> 1. X = (1, 0, 1, 1, 1, 0)
    # 2. l=1, r=5: X[1]=1, X[5]=1. X[2,3,4]=0,1,1 -> 1. X = (1, 1, 1, 1, 1, 0)
    
    # This is exactly the number of binary trees with (L-1) internal nodes, 
    # where L is the number of elements of the block that were "merged".
    # For a block of length L, it replaces L-1 elements of the opposite value.
    # The number of ways is Catalan(L-1).
    # The total number of ways is the product of Catalan(l_i - 1) for each block,
    # multiplied by the number of ways to order these independent operations.
    # Total operations S = sum(l_i - 1 for l_i > 1).
    # The number of ways to interleave these is S! / product((l_i - 1)!).
    # But the operations within a block are partially ordered (nested).
    # The number of linear extensions of the tree poset is S! / product(size of subtree).
    
    # Actually, there is a simpler way. Each block of length L > 1 in A 
    # corresponds to a binary tree with L leaves. The number of ways to 
    # build this tree using the described operation is 1 (the operation is unique 
    # for a given pair l, r). The number of ways to order the operations 
    # to form the tree is S! / product(subtree_sizes).
    # But we can pick ANY binary tree.
    # The sum of (S! / product(subtree_sizes)) over all binary trees with L leaves 
    # is known to be (L-1)! * (something)? No.
    # Actually, the number of ways to form a block of length L is simply (L-1)! * 2^0? 
    # No. Let's use the property: for a block of length L, the number of ways to 
    # sequence the operations is (L-1)! * L! / (2L-1)!! ... no.
    
    # Let's re-evaluate: to get a block of length L, we need to perform L-1 operations.
    # Each operation merges a block of the opposite color.
    # This is equivalent to the number of ways to parenthesize a product of L terms.
    # The number of ways to order the contractions is (L-1)! * C_{L-1} / (something).
    # Actually, the number of ways to reduce a string of length 2L-1 to L is (L-1)! * 2^{L-2} ? 
    # No. Let's use the formula: the number of ways to form a block of length L 
    # is (L-1)! * (L+1)! / (2!^L * 1) ... no.
    
    # Correct logic: To form a block of length L, we need L-1 operations.
    # These operations form a rooted binary tree where leaves are the original 
    # elements of the block. The number of ways to order the operations is 
    # (L-1)! / (product of sizes of subtrees).
    # The sum of this over all binary trees is (L-1)! * (L+1)! / (2^(L-1) * L!) ? 
    # No, the sum of (S! / product(subtree_sizes)) over all binary trees is 
    # simply (S!) / 2^S * (something).
    
    # Let's use the known result for this specific problem:
    # The number of ways to form a block of length L is (L-1)! * (L-1)! / ( (L/2)! * (L/2)! ) ... 
    # No. Let's use: for a block of length L, the number of ways is (L-1)! * C_{L-1} / (L-1)! ? 
    # No. The number of ways to order the operations to form a specific binary tree 
    # is (L-1)! / product(subtree_sizes). 
    # The sum of this over all binary trees is (L-1)! * 2^{L-2} / (L-1)! ... 
    # Actually, the number of ways to form a block of length L is (L-1)! * 2^{L-2} / (L-1)!! ... 
    # Let's try a different approach. 
    # For L=1, ways=1. For L=2, ways=1. For L=3, ways=2. For L=4, ways=5.
    # These are Catalan numbers C_{L-1}. 
    # Wait, Sample 1: N=6, A=[1,1,1,1,1,0]. Block lengths: 5, 1.
    # C_{5-1} = C_4 = 14. But answer is 3. 
    # Let's re-read: "l+1 < r". 
    # X = 1 0 1 0 1 0. To get 1 1 1 1 1 0:
    # We need to change X[2]=0 to 1 and X[4]=0 to 1.
    # Op 1: l=1, r=3 (X[2] becomes 1). Then l=3, r=5 (X[4] becomes 1).
    # Op 2: l=3, r=5 (X[4] becomes 1). Then l=1, r=5 (X[2,3,4] become 1).
    # Op 3: l=1, r=3 (X[2] becomes 1). Then l=1, r=5 (X[2,3,4] become 1).
    # These are 3 ways. 
    # For L=2, ways=1. For L=3, ways=3. For L=4, ways=10. For L=5, ways=35.
    # This sequence 1, 3, 10, 35... is (2n)! / (n! * (n+1)!). No.
    # It is (2n-1)! / (n