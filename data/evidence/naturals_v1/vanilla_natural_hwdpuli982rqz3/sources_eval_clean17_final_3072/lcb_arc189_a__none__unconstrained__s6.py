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
    # Note: The problem says cell i (1 <= i <= N), so X_i = i % 2.
    # In 0-indexed list A, A[i] corresponds to cell i+1.
    # Initial X[i] = (i + 1) % 2.
    X = [(i + 1) % 2 for i in range(N)]

    # Validation: The operation only allows replacing a range with the value of the boundaries.
    # This means we can never change the values of A[0] or A[N-1] from their initials.
    # Also, any contiguous block of identical values in A must have been formed by 
    # expanding from an original cell that had that value.
    if A[0] != X[0] or A[N-1] != X[N-1]:
        print(0)
        return

    # The operation is: choose l, r such that X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This is exactly the process of removing "bubbles" of alternating values.
    # Specifically, if we have a sequence like 1 0 1, we can turn it into 1 1 1.
    # This looks like a grammar reduction or a stack-based matching problem.
    # The target A is reachable if A can be reduced to X by reversing the operation,
    # or X can be reduced to A.
    # Actually, the operation is: find l, r where X[l]==X[r] and the middle is different.
    # This means we are filling in gaps.
    
    # Let's compress A into blocks of identical consecutive elements.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # The only way to get a block of length k > 1 is to use the operation.
    # The operation requires l and r to be the same.
    # If we have a block of 1s in A, it must have originated from at least one '1' in X.
    # Since X is 1 0 1 0..., any block of length k in A covers at least ceil(k/2) elements of X.
    
    # Correct approach:
    # The operation is essentially: if you have 0 1 0, you can make it 0 0 0.
    # This is equivalent to saying: you can delete a contiguous segment of length 1 
    # if it is surrounded by two identical elements.
    # To reach A from X, we must be able to partition A into segments that 
    # "cover" the alternating sequence X.
    # Specifically, a block of identical values of length L in A starting at index i
    # can be formed if the original X sequence in that range [i, i+L-1] 
    # contains at least one instance of that value.
    # But X is 1 0 1 0..., so any range of length >= 1 contains both 0 and 1 
    # unless the range is length 1.
    # Wait, the constraint is: X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This means the middle part must be a single value.
    # Example: 1 0 1 -> 1 1 1. The middle was length 1.
    # To get a block of length 3, we can do: 1 0 1 0 1 -> (op at 0,2) -> 1 1 1 0 1 -> (op at 2,4) -> 1 1 1 1 1.
    
    # Let's simplify: we have a sequence of blocks in A.
    # Let the blocks be (val_1, len_1), (val_2, len_2), ..., (val_m, len_m).
    # For each block j, we need to produce len_j identical elements.
    # The original X in that range is alternating.
    # The number of ways to form a block of length L using this operation is 
    # known to be the Catalan-related number or specifically:
    # If L=1, 1 way. If L=2, 1 way (since X is 1 0 1 0, one of the two must be the target).
    # Actually, the number of ways to form a block of length L is C(L-1, floor((L-1)/2)).
    # No, that's for different problems. 
    # Let's re-evaluate: to turn 1 0 1 0 1 into 1 1 1 1 1:
    # We can pick (0,2) then (2,4) or (2,4) then (0,2). (2 ways)
    # For L=3, X is 1 0 1. Only (0,2) works. (1 way)
    # For L=4, X is 1 0 1 0. We can pick (0,2) -> 1 1 1 0 or (1,3) -> 1 0 0 0. (2 ways)
    # For L=5, X is 1 0 1 0 1. 
    # Ops: (0,2), (2,4) or (2,4), (0,2). Also (0,4) is NOT allowed because middle is not uniform.
    # Wait, (0,4) is allowed if X[1..3] are all different from X[0].
    # But X is 1 0 1 0 1, so X[1]=0, X[2]=1, X[3]=0. Not uniform.
    # So for L=5, we must do (0,2) and (2,4) in any order. (2 ways)
    # For L=6, X is 1 0 1 0 1 0. 
    # We can do (0,2), (2,4), (4,6) in 3! ways? No, (0,2) and (4,6) commute, but (0,2) and (2,4) don't?
    # Actually, they do. If we do (0,2), X becomes 1 1 1 0 1 0. Then (2,4) makes it 1 1 1 1 1 0.
    # If we do (2,4), X becomes 1 0 1 1 1 0. Then (0,2) makes it 1 1 1 1 1 0.
    # The number of ways to form a block of length L is (L // 2)!
    # Let's check Sample 1: N=6, A=[1,1,1,1,1,0]. 
    # Block 1: val=1, len=5. X range [0,4] is 1 0 1 0 1. 
    # Ops: (0,2) and (2,4). These are 2 operations. They can be done in 2! = 2 ways.
    # Block 2: val=0, len=1. 0 ways.
    # Total = 2? Sample says 3. Let me re-read.
    # Sample 1: X = (1, 0, 1, 0, 1, 0). A = (1, 1, 1, 1, 1, 0).
    # Op 1: (2, 4) -> X = (1, 0, 0, 0, 1, 0).
    # Op 2: (1, 5) -> X = (1, 1, 1, 1, 1, 0).
    # This is different! The indices l, r are 1-based. (2, 4) means X[1] and X[3].
    # X[1]=0, X[3]=0. Middle is X[2]=1. Different. Correct.
    # Then X becomes (1, 0, 0, 0, 1, 0).
    # Then (1, 5) means X[0] and X[4]. X[0]=1, X[4]=1. Middle is X[1..3]=(0,0,0). Different. Correct.
    # This means we can nest operations.
    
    # Let f(L) be the number of ways to make a block of length L.
    # For L=1, f(1)=1.
    # For L=2, f(2)=1.
    # For L=3, X is 1 0 1. Op (1,3) -> 1 1 1. f(3)=1.
    # For L=4, X is 1 0 1 0. Op (1,3) -> 1 1 1 0 or (2,4) -> 1 0 0 0. f(4)=2.
    # For L=5, X is 1 0 1 0 1. 
    # 1. Op (1,3) -> 1 1 1 0 1. Then (3,5) -> 1 1 1 1 1.
    # 2. Op (3,5) -> 1 0 1 1 1. Then (1,3) -> 1 1 1 1 1.
    # 3. Op (1,5) -> 1 1 1 1 1. (Since X[1]=1, X[5]=1, and X[2,3,4]=0,1,0... NO)
    # Wait, (1,5) is only allowed if X[2,3,4] are all the same and different from X[1].
    # But X is 1 0 1 0 1. X[2,3,4] is 0, 1, 0. Not the same.
    # So for L=5, f(5)=2.
    # For L=6, X is 1 0 1 0 1 0.
    # We can do (1,3), (3,5) -> 1 1 1 1 1 0. (2 ways)
    # Or (2,4), (4,6) -> 1 0 0 0 0 0. (2 ways)
    # But we want A = 1 1 1 1 1 0. So only the first one. f(6)=2.
    
    # The number of ways to form a block of length L is (L // 2)!
    # Let's check Sample 2: N=10, A=1 1 1 1 1 0 1 1 1 0
    # Blocks: (1, 5), (0, 1), (1, 3), (0, 1)
    # Ways: f(5) * f(1) * f(3) * f(1) = 2! * 1 * 1! * 1 = 2.
    # Sample 2 output is 9. My f(L) is wrong.
    
    # Let's re-examine: l+1 < r, X[l]==X[r], X[i] != X[l] for l < i < r.
    # This is like a parenthesis matching problem.
    # To get a block of length L, we need to perform (L // 2) operations.
    # Each operation consumes one "middle" element of the opposite color.
    # For L=5 (1 0 1 0 1), we need to eliminate two 0s.
    # The 0s are at indices 2 and 4.
    # We can eliminate index 2 using (1, 3) and index 4 using (3, 5).
    # These two operations can be done in any order. 2! = 2.
    # For L=3 (1 0 1), one 0 at index 2. Op (1, 3). 1! = 1.
    # For Sample 2: A = 1 1 1 1 1 0 1 1 1 0
    # X = 1 0 1 0 1 0 1 0 1 0
    # Block 1: A[0..4] = 1 1 1 1 1. X[0..4] = 1 0 1 0 1. 0s at indices 1, 3