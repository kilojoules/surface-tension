```python
import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # Initial state X_i = i % 2 (1-indexed)
    # Note: The problem says cell i has i % 2. 
    # For i=1, X_1=1; i=2, X_2=0; i=3, X_3=1...
    # This is a alternating sequence 1, 0, 1, 0...
    
    # The operation: choose l, r where X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This replaces the block between l and r with X[l].
    # This is essentially merging a block of identical values into a larger block.
    # The only way to change a value is if it's surrounded by the other value.
    # This structure is identical to the problem of counting ways to build a 
    # specific binary string via "range fills" starting from 101010...
    # The key observation is that we can only perform an operation if the 
    # middle elements are different from the endpoints.
    # This means we are collapsing "sandwiches" (0 1 0 -> 0 0 0 or 1 0 1 -> 1 1 1).
    
    # Let's group the target array A into blocks of identical consecutive elements.
    # Example: 1 1 1 1 1 0 -> blocks: (1, length 5), (0, length 1)
    # The initial state is 1 0 1 0 1 0...
    # A block of length L in the target A corresponds to L cells in the initial state.
    # In the initial state, these L cells are alternating.
    # To make them all the same value V, we need to perform operations.
    # If the initial values are already V, they stay. If they are !V, they must be flipped.
    # A block of length L requires (L-1)//2 operations if the endpoints match the target.
    # However, the problem is simpler: we can only perform an operation if the 
    # interior is different. This is exactly the process of removing 
    # "peaks" and "valleys" in a 1D signal.
    
    # The number of ways to reduce a sequence of length L of alternating bits 
    # to a sequence of identical bits is given by the Catalan-like 
    # structure of the operations. Specifically, for a block of length L,
    # the number of ways is the (L-1)-th Catalan number if we view it as 
    # nested parentheses, but the constraint l+1 < r means we need at least 
    # one element in between.
    
    # Correct combinatorial insight:
    # Each block of length L in A that differs from the initial alternating 
    # pattern requires operations. If the block is A_i, A_{i+1}... A_{i+L-1},
    # and it matches the initial pattern, 0 operations.
    # If it's a solid block of length L, the number of ways to form it is 
    # the number of binary trees with L leaves, which is Catalan(L-1).
    # Wait, the condition is l+1 < r. This means the distance is at least 2.
    # For a block of length L, the number of ways to "fill" it is Catalan((L-1)//2).
    # But this only applies if the block's value matches the initial values 
    # at the boundaries of the block.
    
    # Let's re-evaluate:
    # The only way to get a block of L identical values is if the initial 
    # values at the boundaries of the block were already that value.
    # Initial: 1 0 1 0 1 0
    # Target: 1 1 1 1 1 0
    # Block 1: indices 1-5 are '1'. Initial: 1 0 1 0 1.
    # This is a sequence of length 5. The number of ways to make it all '1's
    # is Catalan((5-1)//2) = Catalan(2) = 2.
    # Total ways = Product of Catalan((L_i - 1)//2) for each block i.
    # If L_i is even, it's impossible to make them all the same value 
    # because the endpoints of the block in the initial state would be 
    # different (one 0, one 1), so we can never satisfy X[l] == X[r].
    # UNLESS the block is at the boundary of the array.
    
    # Let's refine:
    # A block of length L starting at index i (1-indexed).
    # Initial values: i%2, (i+1)%2, ..., (i+L-1)%2.
    # Target value: V.
    # For the operation to be possible, we need the endpoints to be V.
    # So i%2 == V and (i+L-1)%2 == V.
    # This implies L must be odd.
    # If L is even, it's impossible, UNLESS the block is "extended" by 
    # an adjacent block of the same value (but blocks are maximal).
    # Actually, if L is even, we can never make them all the same value 
    # using the given operation because the parity of the endpoints 
    # of any range [l, r] in the initial string is different if r-l is odd.
    
    # Special case: if the target A is not reachable, answer is 0.
    # A is reachable if and only if every maximal block of identical values 
    # has a length L such that the initial values at its boundaries 
    # match the target value.
    # Initial X_i = i % 2.
    # For a block from index i to i+L-1 with value V:
    # We need X_i == V and X_{i+L-1} == V.
    # Since X_i = i % 2, this means i % 2 == V and (i+L-1) % 2 == V.
    # This requires (i + L - 1) % 2 == i % 2, so L-1 must be even, so L must be odd.
    
    # Wait, the sample 1: N=6, A=[1,1,1,1,1,0]. 
    # Block 1: indices 1-5, value 1. L=5 (odd). X_1=1, X_5=1. OK.
    # Block 2: index 6, value 0. L=1 (odd). X_6=0. OK.
    # Ways: Catalan((5-1)//2) * Catalan((1-1)//2) = Catalan(2) * Catalan(0) = 2 * 1 = 2.
    # But sample output says 3. Why?
    # Let's re-read: "Choose cells l and r (l+1 < r)".
    # Sample 1: X = (1, 0, 1, 0, 1, 0)
    # Op 1: l=2, r=4. X becomes (1, 0, 0, 0, 1, 0).
    # Op 2: l=1, r=5. X becomes (1, 1, 1, 1, 1, 0).
    # Another way:
    # Op 1: l=1, r=3. X becomes (1, 1, 1, 0, 1, 0).
    # Op 2: l=3, r=5. X becomes (1, 1, 1, 1, 1, 0).
    # Another way:
    # Op 1: l=1, r=5. X becomes (1, 1, 1, 1, 1, 0).
    # Total = 3.
    # This is exactly the number of ways to triangulate a polygon or 
    # the number of binary trees. For L=5, the answer is 3.
    # The number of ways to reduce a block of length L to a single value 
    # is Catalan((L-1)//2) ONLY if we can only pick l, r that are 
    # "currently" the same. But the operation says "replace... with 
    # the integer written in cell l".
    # For L=5, the ways are:
    # 1. (1,5) -> done
    # 2. (1,3) then (3,5) -> done
    # 3. (3,5) then (1,5) -> done
    # 4. (2,4) then (1,5) -> done
    # Wait, the sample says 3. Let's list them:
    # 1. (2,4) then (1,5)
    # 2. (1,3) then (3,5) -- No, (3,5) is only possible if X[3]==X[5].
    # Initial: 1 0 1 0 1 0
    # (1,3) -> 1 1 1 0 1 0. Now X[3]=1 and X[5]=1. So (3,5) is possible.
    # (3,5) -> 1 0 1 1 1 0. Now X[1]=1 and X[3]=1. So (1,3) is possible.
    # (1,5) -> 1 1 1 1 1 0.
    # The 3 ways for L=5 are:
    # - {(2,4), (1,5)}
    # - {(1,3), (3,5)}
    # - {(3,5), (1,3)}
    # Actually, the number of ways to clear a block of length L is 
    # the (L-1)//2-th Catalan number? No.
    # For L=1, ways=1.
    # For L=3, ways=1: {(1,3)}.
    # For L=5, ways=3: {(2,4),(1,5)}, {(1,3),(3,5)}, {(3,5),(1,3)}.
    # This sequence 1, 1, 3, ... looks like it might be related to 
    # the number of ways to parenthesize a product, but the 
    # "middle" must be different.
    # The correct formula for a block of length L is:
    # If L is even, 0.
    # If L is odd, the number of ways is the (L-1)//2-th "Fine number" 
    # or something else? 
    # Let's check L=5 again. The operations are on indices (l, r).
    # The possible (l, r) pairs for L=5 are (1,3), (2,4), (3,5), (1,5).
    # But (1,5) can only be done if the middle is different.
    # Initially: 1 0 1 0 1.
    # If we do (1,5), we are done. (1 way)
    # If we do (1,3), we get 1 1 1 0 1. Then we can do (3,5). (1 way)
    # If we do (3,5), we get 1 0 1 1 1. Then we can do (1,3). (1 way)
    # If we do (2,4), we get 1 0 0 0 1. Then we can do (1,5). (1 way)
    # Total = 4? But the sample says 3.
    # Let's re-read: "The integer written in cell i (l < i < r) is different from the integer written in cell l."
    # For (1,5), the cells 2,3,4 must be different from X[1].
    # Initial: 1 0 1 0 1. X[2]=0, X[3]=1, X[4]=0.
    # X[3] is NOT different from X[1]. So (1,5) is NOT allowed initially.
    # We must first make X[3] different from X[1].
    # To make X[3]=0, we can use (2,4).
    # X becomes 1 0 0 0 1. Now X[2,3,4] are all 0, which is different from X[1]=1.
    # Now (1,5) is allowed.
    # So the only ways are:
    # 1. (2,4) then