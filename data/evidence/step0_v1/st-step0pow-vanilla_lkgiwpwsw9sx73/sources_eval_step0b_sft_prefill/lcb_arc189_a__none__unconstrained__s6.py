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

    # Initial state X_i = i % 2. 
    # Note: The problem says cell i (1 <= i <= N). 
    # In 0-indexed A, A[i] corresponds to cell i+1.
    # Initial X[i] = (i + 1) % 2.
    X = [(i + 1) % 2 for i in range(N)]

    # Validation: If A_i differs from X_i and cannot be changed, 0.
    # The operation only changes cells l+1 to r-1. 
    # A[0] and A[N-1] must match X[0] and X[N-1].
    if A[0] != X[0] or A[N-1] != X[N-1]:
        print(0)
        return

    # The operation is: if X[l] == X[r] and X[i] != X[l] for l < i < r,
    # then X[i] becomes X[l] for l < i < r.
    # This is equivalent to merging adjacent blocks of identical values.
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    
    # To reach state A from X, we must be able to "fill" the blocks of A.
    # X is 1, 0, 1, 0, 1, 0...
    # A block of length k in A starting at index i consists of A[i] repeated k times.
    # In X, the values alternate. To make a block of length k, we need to perform
    # operations. A block of length 1 requires 0 operations.
    # A block of length k > 1 requires (k-1)//2 operations if the boundaries match.
    
    # Specifically, if we have a block of value V from index i to j,
    # it must be that X[i] == V and X[j] == V.
    # If X[i] != V or X[j] != V, it's impossible because the operation 
    # only replaces the interior with the boundary value.
    
    # Let's check if A is reachable from X.
    # X[i] = (i + 1) % 2.
    # For any contiguous segment in A of the same value V, the endpoints must be V in X.
    # Since X alternates, this means the length of any contiguous segment in A must be odd.
    
    # We can use a trick: if we XOR A[i] with (i % 2), the resulting sequence 
    # must be constant within each block of identical elements in A.
    # Actually, simpler: A[i] must be equal to (i + 1) % 2 whenever A[i] != A[i-1].
    # Wait, the constraint is: for a block A[i...j] of value V, 
    # we need X[i] == V and X[j] == V.
    # Since X[i] and X[i+1] are different, this implies (j - i + 1) must be odd.
    
    # Let's group A into blocks of identical elements.
    # blocks = [length1, length2, ...]
    # If any length is even, the answer is 0.
    
    # To calculate the number of ways:
    # For a block of length k (k is odd), the number of ways to form it is 
    # the number of ways to parenthesize the operations.
    # This is the Catalan number C_m where m = (k-1)//2.
    # C_m = (1/(m+1)) * (2m choose m).
    
    # The total number of ways is the product of C_m for each block.
    # However, the operations can be interleaved between different blocks.
    # If we have m1, m2, ..., mk operations, the total ways to order them is
    # (m1 + m2 + ... + mk)! / (m1! * m2! * ... * mk!) * product(C_mi)
    # = (sum m_i)! / product(m_i! * (m_i + 1))
    
    # Let's refine:
    # For a block of length k, let m = (k-1)//2. 
    # The number of ways to collapse the alternating sequence of length k 
    # into a uniform sequence using the described operation is C_m.
    # These m operations are "local" to the block.
    # The total number of operations is M = sum(m_i).
    # The number of ways to sequence these M operations is M! / product(m_i!) 
    # multiplied by the ways to perform them within each block.
    # Total = (M!) / product(m_i!) * product(C_m_i)
    # = (M!) / product(m_i! * (m_i + 1) / m_i!) = M! / product(m_i + 1)
    # Wait, C_m = (2m)! / (m! (m+1)!).
    # Total = (M!) * product( (2m_i)! / (m_i! (m_i+1)!) ) / (M! / product(m_i!))
    # Total = product( (2m_i)! / (m_i! (m_i+1)) ) / (1 / product(m_i!)) ??? No.
    
    # Correct logic:
    # The operations within one block are partially ordered (you must do the 
    # inner ones before the outer ones to satisfy the "different from l" condition).
    # For a block of length k, m = (k-1)//2. The number of valid sequences of 
    # operations to fill the block is C_m.
    # These m operations must be performed in a specific partial order.
    # The number of linear extensions of the product of these partial orders is:
    # M! / product( (number of descendants of each node in the poset) )
    # For the Catalan structure (binary tree), the number of ways to linearize 
    # the construction is M! / product(size of subtree at each node).
    # But the problem is simpler: the operations are nested.
    # To fill a block of length 3 (m=1), 1 way.
    # To fill a block of length 5 (m=2), 2 ways: (inner then outer) or (two separate).
    # Actually, the number of ways to form a block of length k is exactly C_m.
    # And these m operations can be interleaved with operations from other blocks.
    # The number of ways to interleave several sequences of lengths m_i is M! / product(m_i!).
    # However, the C_m ways already account for the internal ordering.
    # The correct formula for the number of ways to perform the operations is:
    # (M!) / product( (m_i + 1)! ) * product( (2m_i)! / m_i! ) ? No.
    
    # Let's re-evaluate:
    # For a single block of length k, m = (k-1)//2. 
    # The operations must be done from the "inside out" or "simultaneously".
    # This is exactly the number of ways to label a binary tree with 1...M 
    # such that a parent is labeled after its children.
    # The number of such ways is M! / product(subtree_size).
    # For a block of length k, the "operation tree" is a structure where 
    # each operation covers a range. To get C_m, the tree is a binary tree.
    # The sum of (M! / product(subtree_size)) over all binary trees is simply 
    # the number of ways to build the sequence.
    # Actually, there is a known result: the number of ways to reduce the 
    # alternating sequence to a constant sequence is (2m)! / (m! * (m+1)!) * m! ? No.
    
    # Let's use the property: to clear a block of length k, we need m = (k-1)//2 operations.
    # These operations can be represented as a binary tree with m nodes.
    # The number of ways to order these operations is m! / (product of subtree sizes).
    # The sum of (m! / product of subtree sizes) over all binary trees is the Catalan number C_m.
    # Wait, the sum of (m! / product of subtree sizes) is actually 1? No.
    # The number of ways to label a specific binary tree is m! / product(subtree_sizes).
    # The sum of this over all binary trees is the number of permutations 
    # that are "heap-ordered".
    # For a fixed m, the sum of (m! / product(subtree_sizes)) over all binary trees 
    # is actually (2m)! / (m! * 2^m) ? No.
    
    # Let's re-read: "Two sequences are different if lengths differ or (l, r) differ".
    # For a block of length k, m = (k-1)//2.
    # The number of ways to form this block is C_m.
    # These m operations must be performed such that if operation A is inside 
    # the range of operation B, A must come before B.
    # This is a poset. The number of linear extensions is m! / product(subtree_size).
    # The total number of ways is:
    # M! * product( sum_{trees T} (1 / product_{node v} size(v)) )
    # The sum sum_{T} (1 / product size(v)) is known to be 1 / m!.
    # Wait, if that's true, the answer is just M!.
    # Let's check Sample 1: N=6, A=[1,1,1,1,1,0]. 
    # X=[1,0,1,0,1,0]. Block 1: length 5, m=2. Block 2: length 1, m=0.
    # M = 2. Answer is 3. 
    # If the formula was M!, 2! = 2. But answer is 3.
    # C_2 = 2. M! / product(m_i!) * product(C_{m_i}) = 2! / 2! * 2 = 2. Still not 3.
    
    # Let's re-calculate for Sample 1:
    # X = 1 0 1 0 1 0
    # Target A = 1 1 1 1 1 0
    # We need to change X[1], X[2], X[3] to 1.
    # Op 1: l=1, r=3 (indices 2, 4). X becomes 1 1 1 0 1 0.
    # Op 2: l=0, r=4 (indices 1, 5). X becomes 1 1 1 1 1 0.
    # OR
    # Op 1: l=3, r=5 (indices 4, 6). X becomes 1 0 1 1 1 0.
    # Op 2: l=0, r=4 (indices 1, 5). X becomes 1 1 1 1 1 0.
    # OR
    # Op 1: l=0, r=4 (indices 1, 5). X becomes 1 1 1 1 1 0.
    # Wait, the condition is: X[i] for l < i < r must be DIFFERENT from X[l].
    # In X = 1 0 1 0 1 0:
    # For l=0, r=4: X[0]=1, X[4]=1. Middle are X[1]=0, X[2]=1, X[3]=0.
    # But X[2] is 1, which is NOT different from X[0]. 
    # So (0, 4) cannot be the first operation.
    # We must first make X[2] different from 1. But X[2] is already 1.
    # The condition is: X[i] != X[l] for all l < i < r.
    # This means the middle part must be a sequence of the opposite value.
    # To turn 1 0 1 0 1 into 1 1 1 1 1:
    # We can do (1, 