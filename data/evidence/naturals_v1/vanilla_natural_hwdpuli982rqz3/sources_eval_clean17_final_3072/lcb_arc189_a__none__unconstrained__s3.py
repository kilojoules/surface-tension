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
    # was different. This is essentially a way to merge adjacent blocks of 
    # the same value.
    # Specifically, if we have a target sequence A, we can view it as a 
    # sequence of blocks of identical values.
    # Let the compressed sequence be B. If A = (1, 1, 1, 0, 0, 1), B = (1, 0, 1).
    # The operation allows us to reduce the number of blocks.
    # To get from (1, 0, 1, 0, ...) to A, we must perform operations that 
    # "overwrite" the alternating pattern.
    
    # Let's analyze the structure:
    # The initial state is X_i = i % 2.
    # The operation (l, r) is valid if X_l == X_r and X_i != X_l for l < i < r.
    # This means the segment [l, r] must look like (v, 1-v, v) or (v, 1-v, 1-v, ..., 1-v, v).
    # After the operation, it becomes (v, v, ..., v).
    # This is equivalent to saying we can remove a block of identical values 
    # if it is flanked by two blocks of the opposite value.
    
    # Let the target sequence A be represented as blocks of lengths L_1, L_2, ..., L_k.
    # The initial sequence is 1, 0, 1, 0... (or 0, 1, 0, 1...).
    # Note: The problem says cell i has i % 2. So X_1=1, X_2=0, X_3=1...
    # Wait, i mod 2 for i=1 is 1, i=2 is 0. So X = [1, 0, 1, 0, ...].
    
    # If A_i != X_i for some i, that cell must have been changed by an operation.
    # An operation (l, r) changes X_{l+1}...X_{r-1} to X_l.
    # This is only possible if X_l == X_r and X_{l+1}...X_{r-1} were all different from X_l.
    # Since X is alternating, this means r-l must be 2.
    # (l, l+2) changes X_{l+1} to X_l. Now X_l, X_{l+1}, X_{l+2} are all the same.
    # This creates a block of 3 identical values.
    
    # Key insight: This problem is equivalent to counting ways to parenthesize 
    # the reduction of a string of alternating characters to the target string A.
    # Each operation (l, r) effectively removes a "peak" or "valley" in the 
    # alternating sequence.
    # If we have a block of length L in A, it corresponds to a sequence of 
    # operations that merged L-1 alternating elements.
    # The number of ways to form a block of length L using these operations is 
    # the (L-1)-th Catalan number? No, it's simpler.
    # For a block of length L, the number of ways to build it is C_{L-1} where 
    # C is the Catalan number? Let's check Sample 1: N=6, A=[1,1,1,1,1,0].
    # X=[1,0,1,0,1,0]. Target A has a block of five 1s and one 0.
    # To get five 1s from [1,0,1,0,1], we need 2 operations.
    # Op 1: (1, 3) -> [1,1,1,0,1] then Op 2: (3, 5) -> [1,1,1,1,1]
    # Op 1: (3, 5) -> [1,0,1,1,1] then Op 2: (1, 3) -> [1,1,1,1,1]
    # Op 1: (1, 5) -> [1,1,1,1,1] (since X_1=1, X_5=1 and X_2,3,4 are not all 1? No, X_2=0, X_3=1, X_4=0. 
    # The condition is X_i != X_l for l < i < r. So X_2, X_3, X_4 must all be 0.
    # But X is [1, 0, 1, 0, 1]. X_3 is 1. So (1, 5) is NOT allowed initially.)
    
    # Correct logic: To merge a segment into one color, we must eliminate the 
    # opposite color blocks inside it. Each operation removes one block of 
    # opposite color and merges two blocks of the same color.
    # If we have a block of length L in A, it means we started with a sequence 
    # of length 2L-1 (alternating) and performed L-1 operations.
    # The number of ways to do this is the Catalan number C_{L-1}.
    # Total ways = Product of C_{L_i - 1} for all blocks i, but we must also 
    # consider the order of operations between different blocks.
    # Actually, the operations for different blocks are independent.
    # The total number of operations is sum(L_i - 1).
    # The number of ways to interleave these sequences of operations is 
    # (Total Ops)! / Product((L_i - 1)!).
    # But the operations within one block must follow the Catalan structure.
    # The number of ways to reduce a block of length L is C_{L-1}.
    # Total = (Sum(L_i-1))! / Product((L_i-1)!) * Product(C_{L_i-1})
    # Since C_n = (2n)! / ((n+1)! n!), this simplifies to:
    # Total = (Sum(L_i-1))! * Product( (2(L_i-1))! / ((L_i)! (L_i-1)!) ) / Product((L_i-1)!)
    # This is getting complex. Let's use the property:
    # A block of length L is formed by L-1 operations. The number of ways is C_{L-1}.
    # These operations are partially ordered. The total number of linear extensions 
    # of the combined poset is (Total Ops)! / Product( (L_i-1) + 1 ) ? No.
    # The correct formula for the number of ways to form the blocks is:
    # (Total Ops)! / Product( (2*(L_i-1)) / (L_i) ) ... no.
    
    # Let's use the known result for this specific problem:
    # The answer is (Total Ops)! / Product( (L_i * (L_i + 1) // 2) ) ... no.
    # The number of ways to form a block of length L is C_{L-1}.
    # The number of ways to interleave these is (Total Ops)! / Product( (L_i-1)! ) 
    # BUT the operations within a block are not totally ordered.
    # The actual answer is: (Total Ops)! / Product( (L_i * (L_i + 1) // 2) ) is for a different problem.
    # For this problem, the number of ways to form a block of length L is 1 if we 
    # only allow (l, l+2) operations. But we can pick any l, r.
    # The number of ways to reduce a block of length L is (2^(L-1) - 1)? No.
    
    # Let's re-evaluate: an operation (l, r) is possible if X_l == X_r and X_{l+1}...X_{r-1} are all the same (and different from X_l).
    # This means we are replacing a block of length (r-l-1) with the color of the boundaries.
    # To get a block of length L, we must have started with a sequence of length 2L-1.
    # The number of ways to do this is exactly 1 if we only consider the "structure".
    # But the operations are labeled by (l, r).
    # For a block of length L, the number of ways to form it is (L-1)! * 2^{L-2} ? No.
    # Let's test Sample 1: N=6, A=[1,1,1,1,1,0]. Blocks: L1=5, L2=1.
    # Total Ops = (5-1) + (1-1) = 4.
    # Sample 1 Output is 3. For L=5, the number of ways is 3?
    # If L=3, X=[1,0,1], Op(1,3) -> [1,1,1]. 1 way.
    # If L=4, X=[1,0,1,0], not possible to get [1,1,1,1] because X_4=0.
    # Wait, the initial X is X_i = i % 2.
    # Sample 1: X = [1, 0, 1, 0, 1, 0]. A = [1, 1, 1, 1, 1, 0].
    # To get A, we need X_2, X_3, X_4, X_5 to become 1.
    # X_1=1, X_3=1, X_5=1.
    # Possible sequences:
    # 1. (1, 3) -> [1, 1, 1, 0, 1, 0], then (3, 5) -> [1, 1, 1, 1, 1, 0]
    # 2. (3, 5) -> [1, 0, 1, 1, 1, 0], then (1, 3) -> [1, 1, 1, 1, 1, 0]
    # 3. (1, 5) -> [1, 1, 1, 1, 1, 0] (Valid because X_1=1, X_5=1 and X_2,3,4 are not all 1? No, the condition is X_i != X_l for l < i < r. X_2=0, X_3=1, X_4=0. X_3 is 1, which is equal to X_1. So (1, 5) is NOT allowed.)
    # Wait, the sample says 3 ways. Let me re-read.
    # "Choose cells l and r (l+1 < r) ... replace each of the integers in l+1...r-1 with X_l. 
    # Conditions: X_l == X_r AND X_i != X_l for l < i < r."
    # Sample 1: X = [1, 0, 1, 0, 1, 0]. A = [1, 1, 1, 1, 1, 0].
    # Op 1: (1, 3) -> X becomes [1, 1, 1, 0, 1, 0].
    # Now we can do (3, 5) -> [1, 1, 1, 1, 1, 0].
    # Or Op 1: (3, 5) -> X becomes [1, 0, 1, 1, 1, 0].
    # Now we can do (1, 3) -> [1, 1, 1, 1, 1, 0].
    # Or Op 1: (1, 5) -> X becomes [1, 1, 1, 1, 1, 0].
    # Is (1, 5) allowed? X_1=1, X_5=1. X_2=0, X_3=1, X_4=0.
    # The condition is: X_i != X_l for l < i < r.
    # X_3 is 1, and X_1 is 1. So X_3 == X_1. The condition is violated.
    # So (1, 5) is NOT allowed.
    # Then where does the 3rd way come from?
    