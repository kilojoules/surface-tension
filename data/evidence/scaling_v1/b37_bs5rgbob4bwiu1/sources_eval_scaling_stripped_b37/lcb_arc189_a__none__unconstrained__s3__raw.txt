```python
import sys
from functools import reduce

def solve():
    # Read N and the sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # The initial state is X_i = i % 2 (1-indexed).
    # So X = [1, 0, 1, 0, 1, 0, ...]
    # The operation: if X[l] == X[r] and for all l < i < r, X[i] != X[l],
    # then set X[i] = X[l] for l < i < r.
    # This is essentially a range-fill operation that can only happen if 
    # the range is currently alternating and bounded by the same value.
    # This structure describes a process of merging blocks of identical values.
    # A block of identical values A_i...A_j can be formed if it was 
    # originally alternating and we performed operations to fill it.
    # The number of ways to form a block of length 'len' using these operations
    # is given by the (len-1)-th Catalan number if we view it as a 
    # triangulation/parenthesization problem, but specifically here,
    # the number of ways to reduce a sequence of length k to a single value
    # via these specific rules is C_{k-1} where C is the Catalan number.
    # Wait, the rule is: l+1 < r, X[l]==X[r], and X[i] != X[l] for l < i < r.
    # This means we can only collapse a segment if it is exactly "0 1 0" or "1 0 1".
    # This is exactly the structure of a binary tree/parenthesization.
    # For a block of length 'k', the number of ways to form it is Catalan(k-1).
    # However, we must check if the target A is reachable.
    # A is reachable if for every block of identical values of length k,
    # the original values in those positions were alternating.
    # Since the original is 1, 0, 1, 0..., any range always alternates.
    # The only constraint is that we cannot change the values of the endpoints
    # of the total array if they don't match the original X_i.
    # Actually, the operation only changes i for l < i < r. 
    # So X[1] and X[N] can never change.
    # Check if A[i] == (i+1)%2 for i=0 and i=N-1.
    # Wait, the problem says cell i has i % 2. 
    # For i=1: 1%2 = 1. For i=2: 2%2 = 0.
    # So X = [1, 0, 1, 0, ...]
    # A[0] must be 1 and A[N-1] must be (N % 2).
    
    if A[0] != 1 or A[N-1] != (N % 2):
        print(0)
        return

    # Group A into blocks of identical consecutive values
    # Example: 1 1 1 1 1 0 -> blocks of length 5 and 1
    blocks = []
    if N > 0:
        current_val = A[0]
        count = 0
        for val in A:
            if val == current_val:
                count += 1
            else:
                blocks.append(count)
                current_val = val
                count = 1
        blocks.append(count)

    # For each block of length k, there are Catalan(k-1) ways to form it.
    # The total ways is the product of Catalan(k-1) for all blocks.
    # Catalan(n) = (2n)! / ((n+1)! n!)
    
    MOD = 998244353
    
    def nCr_mod(n, r, fact, inv):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv[r] * inv[n - r]) % MOD
        return (num * den) % MOD

    def catalan(n, fact, inv):
        return nCr_mod(2 * n, n, fact, inv) * pow(n + 1, MOD - 2, MOD) % MOD

    # Max n needed for Catalan(k-1) is 2*(N-1)
    # Since we need Catalan(k-1), and sum(k) = N, the max k is N.
    # We need factorials up to 2*N.
    max_fact = 2 * N + 2
    fact = [1] * (max_fact + 1)
    for i in range(2, max_fact + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    
    inv = [1] * (max_fact + 1)
    inv[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv[i] = (inv[i + 1] * (i + 1)) % MOD

    # The number of ways to form a block of length k is Catalan(k-1).
    # If k=1, Catalan(0) = 1.
    # The total number of ways is the product of Catalan(k-1) for all blocks.
    # But there is a catch: the blocks must be reducible.
    # A block of length k is reducible if the original values were alternating.
    # They always are. The only requirement is that the operation 
    # requires l and r to have the same value.
    # In a block of length k, we have k positions. 
    # The original values were X, X^1, X, X^1...
    # To merge them into one value, we need the endpoints to be the same.
    # This means the length of the segment being collapsed must be odd.
    # If we have a block of length k, it means we collapsed k-2 elements 
    # between two indices. The distance between l and r is r - l.
    # The number of elements between them is (r - l - 1).
    # For the operation to be valid, X[l] must equal X[r].
    # Since X is alternating, X[l] == X[r] iff (r - l) is even.
    # This means the number of elements replaced (r - l - 1) must be odd.
    # A block of length k is formed by replacing k-2 elements? No.
    # A block of length k means k consecutive cells have the same value.
    # This is only possible if the original parity allowed it.
    # Actually, the only way to get k identical values is if we started with
    # alternating values and used the operation.
    # The operation replaces (r-l-1) elements. 
    # To turn a segment of length k into identical values, 
    # we need k to be odd? No, let's trace:
    # (1, 0, 1) -> l=1, r=3. X[1]=1, X[3]=1. Replace X[2] with 1. Result: (1, 1, 1).
    # Length 3 becomes identical.
    # (1, 0, 1, 0, 1) -> l=1, r=5. Result: (1, 1, 1, 1, 1).
    # Length 5 becomes identical.
    # What about length 2? (1, 0). Cannot be made (1, 1) or (0, 0) because
    # we need l and r such that l+1 < r. For length 2, there is no such l, r.
    # So blocks of length k can only be formed if k is odd.
    # Wait, Sample 1: 1 1 1 1 1 0. Blocks: 5, 1. Both odd.
    # Sample 2: 1 1 1 1 1 0 1 1 1 0. Blocks: 5, 1, 3, 1. All odd.
    # If any block length is even, it's impossible.
    
    # Let's check the parity of block lengths.
    # A block of length k is possible iff k is odd.
    # If k is even, the answer is 0.
    
    # For odd k, the number of ways to form it is Catalan((k-1)//2).
    # Let's verify: k=3 -> Cat(1)=1. k=5 -> Cat(2)=2.
    # Sample 1: k=5, k=1. Cat(2)*Cat(0) = 2 * 1 = 2. 
    # But sample output says 3. Let me re-read.
    # "Choose cells 2 and 4. X becomes (1, 0, 0, 0, 1, 0). Then 1 and 5."
    # Initial: 1 0 1 0 1 0
    # Op 1: l=2, r=4. X[2]=0, X[4]=0. X[3] becomes 0. X: 1 0 0 0 1 0
    # Op 2: l=1, r=5. X[1]=1, X[5]=1. X[2,3,4] become 1. X: 1 1 1 1 1 0
    # This means blocks can be built hierarchically.
    # The number of ways to reduce a segment of length k to a single value
    # is the number of ways to parse a string of length k with a binary 
    # operation, which is Catalan(k-1) if we can combine any two.
    # But here we can only combine if the endpoints are the same.
    # This is exactly the condition for the "Interval DP" or "Matrix Chain"
    # style reduction. For a segment of length k, the number of ways is
    # the (k-1)-th Catalan number? No, let's check k=3: Cat(2)=2? 
    # For k=3 (1,0,1), only one way: l=1, r=3.
    # For k=5 (1,0,1,0,1):
    # 1. l=1, r=5 -> (1,1,1,1,1)
    # 2. l=2, r=4 -> (1,0,0,0,1), then l=1, r=5 -> (1,1,1,1,1)
    # 3. l=1, r=3 -> (1,1,1,0,1), then l=1, r=5 -> (1,1,1,1,1)
    # Total 3 ways. This is the 3rd Fibonacci-like or something?
    # Actually, these are the Motzkin numbers? No.
    # Let's see: k=1: 1 way. k=3: 1 way. k=5: 3 ways.
    # This sequence (1, 1, 3, ...) for k=1, 3, 5...
    # The number of ways to reduce a segment of length 2m+1 is 
    # the m-th Catalan number? Cat(0)=1, Cat(1)=1, Cat(2)=2. 
    # Still not 3.
    # Let's re-evaluate k=5:
    # Ways: {(1,5)}, {(2,4), (1,5)}, {(1,3), (1,5)}.
    # Wait, (1,3) then (3,5) is not possible because X[3] becomes 1, 
    # then X[3]=1 and X[5]=1, so we can do (3,5).
    # So {(1,3), (3,5)} is also a way.
    # Total ways for k=5: {(1,5)}, {(2,4), (1,5)}, {(1,3), (1,5)}, {(3,5), (1,5)}, {(1,3), (3,5), (1,5)}...
    # No, the rule is: l+1 < r, X[l]==X[r], and X[i] != X[l] for l < i < r.
    # For k=5: (1,0,1,0,1)
    # - (1,5) -> (1,1,1,1,1). Done.
    # - (1,3) -> (1,1,1,0,1). Now l=1, r=5 is